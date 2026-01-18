import numpy as np
import streamlit as st
import tensorflow as tf
from transformers import (
    TFAutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
import os
import logging
from pathlib import Path
import markdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "gpt2-medium"
MODEL_PATH = Path("/home/lloyd/Downloads/local_model_store/gpt2-medium")
SAVE_PATH = Path("/home/lloyd/Downloads/local_model_store/gpt2-adaptive")

# Ensure model and tokenizer are available locally
if not MODEL_PATH.exists():
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.save_pretrained(MODEL_PATH)
    model.save_pretrained(MODEL_PATH)
    logging.info("Model and tokenizer downloaded and saved.")
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    logging.info("Model and tokenizer loaded from local storage.")


@st.cache(allow_output_mutation=True)
def load_and_prepare_model(model_path: str) -> tuple:
    """
    Loads and prepares a GPT-2 model from Hugging Face Transformers.

    Args:
        model_path (str): Path to the model directory.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = TFAutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer


def convert_hf_to_tf(hf_model: tf.keras.Model) -> tf.keras.Model:
    """
    Converts a Hugging Face model to a TensorFlow model.

    Args:
        hf_model (tf.keras.Model): The Hugging Face model to convert.

    Returns:
        tf.keras.Model: The converted TensorFlow model.
    """
    tf_model = tf.keras.models.clone_model(hf_model)
    tf_model.trainable = True
    return tf_model


def train_model(
    model: tf.keras.Model, dataset: list, epochs: int = 1
) -> tf.keras.Model:
    """
    Trains the model on the provided dataset for a given number of epochs.

    Args:
        model (tf.keras.Model): The model to be trained.
        dataset (list): A list of tuples (input_text, target_text).
        epochs (int): The number of epochs for training.

    Returns:
        tf.keras.Model: The trained model.
    """
    for epoch in range(epochs):
        for input_text, target_text in dataset:
            loss = model.train_on_batch(input_text, target_text)
            logging.info(f"Epoch {epoch}, Loss: {loss}")
    return model


def preprocess_text(text: str) -> np.ndarray:
    """
    Preprocesses the input text by tokenizing it.

    Args:
        text (str): The input text.

    Returns:
        np.ndarray: The tokenized input.
    """
    return tokenizer.encode(text, return_tensors="np")


def postprocess_response(predictions: np.ndarray) -> str:
    """
    Postprocesses the model predictions by decoding the tokens.

    Args:
        predictions (np.ndarray): The predictions array.

    Returns:
        str: The decoded response.
    """
    return tokenizer.decode(predictions[0])


def perform_tfidf_analysis(text_data: list) -> np.ndarray:
    """
    Performs TF-IDF analysis on the input text data.

    Args:
        text_data (list): A list of text documents.

    Returns:
        np.ndarray: The TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix.toarray()


def perform_kmeans_clustering(tfidf_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Performs K-means clustering on the TF-IDF matrix.

    Args:
        tfidf_matrix (np.ndarray): The TF-IDF matrix.
        n_clusters (int): The number of clusters.

    Returns:
        np.ndarray: The cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(tfidf_matrix)
    return labels


def evaluate_clustering(tfidf_matrix: np.ndarray, labels: np.ndarray) -> float:
    """
    Evaluates the quality of the clustering using silhouette score.

    Args:
        tfidf_matrix (np.ndarray): The TF-IDF matrix.
        labels (np.ndarray): The cluster labels.

    Returns:
        float: The silhouette score.
    """
    score = silhouette_score(tfidf_matrix, labels)
    return score


def perform_pca(tfidf_matrix: np.ndarray, n_components: int) -> np.ndarray:
    """
    Performs Principal Component Analysis (PCA) on the TF-IDF matrix.

    Args:
        tfidf_matrix (np.ndarray): The TF-IDF matrix.
        n_components (int): The number of principal components.

    Returns:
        np.ndarray: The transformed matrix.
    """
    pca = PCA(n_components=n_components)
    transformed_matrix = pca.fit_transform(tfidf_matrix)
    return transformed_matrix


def perform_naive_bayes_classification(text_data: list, labels: list) -> MultinomialNB:
    """
    Performs Naive Bayes classification on the text data.

    Args:
        text_data (list): A list of text documents.
        labels (list): The corresponding labels for the text documents.

    Returns:
        MultinomialNB: The trained Naive Bayes classifier.
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text_data)
    clf = MultinomialNB()
    clf.fit(X, labels)
    return clf


def add_new_layer(model: tf.keras.Model, units: int) -> tf.keras.Model:
    """
    Adds a new dense layer to the model as a form of "neurogenesis".

    Args:
        model (tf.keras.Model): The model to add the layer to.
        units (int): The number of units in the new layer.

    Returns:
        tf.keras.Model: The updated model with the new layer.
    """
    new_layer = tf.keras.layers.Dense(units, activation="relu")
    model.add(new_layer)
    return model


# Load model and tokenizer
model, tokenizer = load_and_prepare_model(str(MODEL_PATH))

# Convert Hugging Face model to TensorFlow
tf_model = convert_hf_to_tf(model)

# Streamlit app setup
st.title("Adaptive Conversational AI")

# Markdown file ingestion for training
uploaded_files = st.file_uploader(
    "Upload conversation files", accept_multiple_files=True, type=["md"]
)
if uploaded_files:
    text_data = []
    for uploaded_file in uploaded_files:
        content = markdown.markdown(uploaded_file.getvalue().decode())
        text_data.append(content)
        processed_content = preprocess_text(content)
        tf_model = train_model(
            tf_model, [(processed_content, processed_content)], epochs=1
        )

    # Perform TF-IDF analysis
    tfidf_matrix = perform_tfidf_analysis(text_data)

    # Perform K-means clustering
    n_clusters = 5
    labels = perform_kmeans_clustering(tfidf_matrix, n_clusters)

    # Evaluate clustering quality
    silhouette_score = evaluate_clustering(tfidf_matrix, labels)
    logging.info(f"Silhouette Score: {silhouette_score}")

    # Perform PCA
    n_components = 2
    pca_matrix = perform_pca(tfidf_matrix, n_components)

    # Perform Naive Bayes classification
    clf = perform_naive_bayes_classification(text_data, labels)

    # Add new layer to the model
    tf_model = add_new_layer(tf_model, units=128)

user_input = st.text_input("Talk to the AI:")
if user_input:
    input_processed = preprocess_text(user_input)
    predictions = tf_model.predict(input_processed)
    response = postprocess_response(predictions)
    st.write(response)
    sentiment = sentiment_score(response)
    entities = extract_entities(response)
    st.write(f"Sentiment Score: {sentiment}, Entities: {entities}")
    feedback = st.text_input("Feedback to improve AI:")
    if feedback:
        feedback_processed = preprocess_text(feedback)
        tf_model = train_model(
            tf_model, [(input_processed, feedback_processed)], epochs=1
        )

# Save the adapted model
tf_model.save_pretrained(SAVE_PATH)
