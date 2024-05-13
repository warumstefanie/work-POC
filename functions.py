import os
from openai import OpenAI
import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial import distance
model = "gpt-4-turbo"
client = OpenAI(api_key=os.environ["OPENAI"])


# Function to create embeddings from text
def create_embeddings(texts):
  response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=texts
  )
  response_dict = response.model_dump()
  
  return [data['embedding'] for data in response_dict['data']]


# Function to create and find distances
def find_n_closest(query_vector, embeddings, n=3):
  distances = []
  for index, embedding in enumerate(embeddings):
    # Calculate the cosine distance between the query vector and embedding
    dist = distance.cosine(query_vector, embedding)
    # Append the distance and index to distances
    distances.append({"distance": dist, "index": index})
  # Sort distances by the distance key
  distances_sorted = sorted(distances, key=lambda x: x["distance"])
  # Return the first n elements in distances_sorted
  return distances_sorted[0:n]



#Function to convert structured text into string
def create_product_text(product):
  return f"""Title: {product["title"]}
Description: {product["short_description"]}
Category: {product["category"]}
Features: {", ".join(product["features"])}"""


#Function to create recommendations
def recommend_products(user_history, products):
    # Create text descriptions of each product in the user's history and compute their embeddings
    history_texts = [create_product_text(product) for product in user_history]
    history_embeddings = create_embeddings(history_texts)
    mean_history_embeddings = np.mean(history_embeddings, axis=0)

    # Filter out products that are already in the user's history
    products_filtered = [product for product in products if product not in user_history]

    # Create text descriptions of the filtered products and compute their embeddings
    product_texts = [create_product_text(product) for product in products_filtered]
    product_embeddings = create_embeddings(product_texts)

    # Find products closest to the mean of history embeddings
    hits = find_n_closest(mean_history_embeddings, product_embeddings)

    # Print titles and distances of recommended products
    recommended_products = [(products_filtered[hit['index']]['title'], hit['distance']) for hit in hits]
    for title, distance in recommended_products:
        print(f"Product: {title}, Distance: {distance}")



