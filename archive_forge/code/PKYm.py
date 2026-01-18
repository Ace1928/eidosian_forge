from transformers import AutoConfig, AutoModel
import os


def load_custom_model(model_directory: str):
    """
    Loads a custom model from a specified directory using Hugging Face's transformers.

    Args:
        model_directory (str): The directory where the custom model and its configuration are saved.

    Returns:
        A model object loaded from the directory.

    Raises:
        FileNotFoundError: If the model directory does not exist.
        Exception: If there are issues loading the model from the directory.
    """
    # Check if the model directory exists
    if not os.path.exists(model_directory):
        raise FileNotFoundError(
            f"The specified model directory does not exist: {model_directory}"
        )

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
MODEL_DIRECTORY = "/RWKV-Runner/models/rwkv-final-v6-1b5"

# Load the model
custom_model = load_custom_model(MODEL_DIRECTORY)

# At this point, `custom_model` can be used for inference or further processing.
