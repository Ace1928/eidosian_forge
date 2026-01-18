import numpy as np
import streamlit as st
import tensorflow as tf
from transformers import (
    TFAutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
import tensorflow_model_optimization as tfmot
import os
import logging
from pathlib import Path
import markdown
from sentiment_analysis import sentiment_score
from ner_analysis import extract_entities

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "gpt2-medium"
MODEL_PATH = Path("/home/lloyd/Downloads/local_model_store/gpt2-medium")
SAVE_PATH = Path("/home/lloyd/Downloads/local_model_store/gpt2-adaptive")

# Pruning and quantization functions
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

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


def apply_model_optimizations(model: tf.keras.Model) -> tf.keras.Model:
    """
    Applies pruning and quantization to the TensorFlow model.

    Args:
        model (tf.keras.Model): The model to optimize.

    Returns:
        tf.keras.Model: The optimized model.
    """
    # Pruning parameters
    pruning_params = {
        "pruning_schedule": tf.keras.optimizers.schedules.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=0.5, begin_step=2000, end_step=10000
        )
    }
    # Apply pruning
    model = prune_low_magnitude(model, **pruning_params)
    # Apply quantization
    quantize_aware_model = tfmot.quantization.keras.quantize_model(model)
    return quantize_aware_model


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


# Load model and tokenizer
model, tokenizer = load_and_prepare_model(str(MODEL_PATH))

# Convert Hugging Face model to TensorFlow
tf_model = convert_hf_to_tf(model)

# Apply optimizations
optimized_model = apply_model_optimizations(tf_model)

# Streamlit app setup
st.title("Adaptive Conversational AI")

# Markdown file ingestion for training
uploaded_files = st.file_uploader(
    "Upload conversation files", accept_multiple_files=True, type=["md"]
)
if uploaded_files:
    for uploaded_file in uploaded_files:
        content = markdown.markdown(uploaded_file.getvalue().decode())
        processed_content = preprocess_text(content)
        optimized_model = train_model(
            optimized_model, [(processed_content, processed_content)], epochs=1
        )

user_input = st.text_input("Talk to the AI:")
if user_input:
    input_processed = preprocess_text(user_input)
    predictions = optimized_model.predict(input_processed)
    response = postprocess_response(predictions)
    st.write(response)
    sentiment = sentiment_score(response)
    entities = extract_entities(response)
    st.write(f"Sentiment Score: {sentiment}, Entities: {entities}")
    feedback = st.text_input("Feedback to improve AI:")
    if feedback:
        feedback_processed = preprocess_text(feedback)
        optimized_model = train_model(
            optimized_model, [(input_processed, feedback_processed)], epochs=1
        )

# Save the adapted model
optimized_model.save_pretrained(SAVE_PATH)
