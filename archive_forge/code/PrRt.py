import numpy as np
import streamlit as st
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer, AutoConfig
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

model_name = "microsoft/Phi-3-mini-128k-instruct"
model_path = (
    "/home/lloyd/Downloads/local_model_store/microsoft/Phi-3-mini-128k-instruct"
)


@st.cache(allow_output_mutation=True)
def load_and_prepare_model(model_path: str) -> tuple:
    """
    Loads and prepares a causal language model from Hugging Face Transformers.

    Args:
        model_path (str): Path to the model on local disk.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Load the model with the custom configuration
    config = Phi3Config.from_pretrained(model_path)
    model = TFAutoModelForCausalLM.from_config(config)
    return model, tokenizer


def convert_hf_to_tf(hf_model):
    """
    Converts a Hugging Face model to a TensorFlow model.

    Args:
        hf_model: The Hugging Face model to convert.

    Returns:
        The converted TensorFlow model.
    """
    tf_model = tf.keras.models.clone_model(hf_model)
    tf_model.trainable = True
    return tf_model


def prune_model(model):
    """
    Applies pruning to the model.

    Args:
        model: The model to be pruned.

    Returns:
        The pruned model.
    """
    pruning_params = {
        "pruning_schedule": tf.keras.optimizers.schedules.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=2000,
            end_step=10000,
        )
    }
    pruned_model = tf.keras.models.clone_model(model)
    pruned_model = prune_low_magnitude(pruned_model, **pruning_params)
    return pruned_model


def quantize_model(model):
    """
    Applies quantization to the model.

    Args:
        model: The model to be quantized.

    Returns:
        The quantized model.
    """
    quantized_model = tf.keras.models.clone_model(model)
    quantized_model = quantize_model(quantized_model)
    return quantized_model


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
            print(f"Epoch {epoch}, Loss: {loss}")
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
model, tokenizer = load_and_prepare_model(model_path)

# Convert Hugging Face model to TensorFlow
tf_model = convert_hf_to_tf(model)

# Apply pruning
pruned_model = prune_model(tf_model)

# Apply quantization
quantized_model = quantize_model(pruned_model)

# Streamlit app setup
st.title("Adaptive Conversational AI")

user_input = st.text_input("Talk to the AI:")
if user_input:
    input_processed = preprocess_text(user_input)
    predictions = quantized_model.predict(input_processed)
    response = postprocess_response(predictions)
    st.write(response)
    feedback = st.text_input("Feedback to improve AI:")
    if feedback:
        feedback_processed = preprocess_text(feedback)
        quantized_model = train_model(
            quantized_model, [(input_processed, feedback_processed)], epochs=1
        )

# Save the adapted model
SAVE_PATH = "/home/lloyd/Downloads/local_model_store/Phi-3-adaptive"
quantized_model.save_pretrained(SAVE_PATH)
