from transformers import AutoConfig, AutoModel
import os


def load_custom_model(model_directory: str):
    """
    Loads a custom model from a specified directory using Hugging Face's transformers.
    If the directory does not exist, it creates the directory.

    Args:
        model_directory (str): The directory where the custom model and its configuration are saved.
                               This can be a relative or absolute path.

    Returns:
        A model object loaded from the directory.

    Raises:
        Exception: If there are issues loading the model from the directory.
    """
    # Construct the absolute path from the relative path
    model_directory = os.path.abspath(model_directory)

    # Check if the model directory exists, if not, create it
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        print(f"Model directory created at: {model_directory}")

    try:
        # Load the configuration from the model directory
        config = AutoConfig.from_pretrained(model_directory)

        # Load the model from the model directory with the loaded configuration
        model = AutoModel.from_pretrained(model_directory, config=config)

        print(f"Model successfully loaded from {model_directory}")
        return model
    except Exception as e:
        raise Exception(
            f"Failed to load the model from the directory: {model_directory}. Error: {e}"
        )


# Specify the directory where your model is saved
MODEL_DIRECTORY = (
    "/home/lloyd/Dropbox/evie_env/cursor/ChatRWKV/ChatRWKV/models/rwkv-final-v6-1b5"
)

# Load the model
custom_model = load_custom_model(MODEL_DIRECTORY)

# At this point, `custom_model` can be used for inference or further processing.
