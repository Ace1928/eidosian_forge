import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Callable, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
import signal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
MD_DIRECTORY: str = "/home/lloyd/Downloads/gpt_chats"
PROCESSED_DATA_DIR: str = "/media/lloyd/Aurora_M2/dandata/processed_data/"
SYNTHETIC_DATA_DIR: str = "/media/lloyd/Aurora_M2/dandata/synthetic_data/"
UNTRAINED_MODEL_PATH: str = "/media/lloyd/Aurora_M2/dandata/untrained_dan_model.pkl"
REGRESSION_TRAINED_MODEL_PATH: str = (
    "/media/lloyd/Aurora_M2/dandata/regression_trained_dan_model.pkl"
)
TEXT_TRAINED_MODEL_PATH: str = (
    "/media/lloyd/Aurora_M2/dandata/text_trained_dan_model.pkl"
)
OUTPUT_DIR: str = "/media/lloyd/Aurora_M2/dandata/outputs/"
INPUT_SIZE: int = 1024
OUTPUT_SIZE: int = 128
SYNTHETIC_DATA_SIZE: int = 1000000
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
LR: float = 0.001
WEIGHT_DECAY: float = 0.0001
SCALE_OUTPUT: bool = True
VECTORIZER: TfidfVectorizer = TfidfVectorizer(max_features=INPUT_SIZE)
BATCH_SIZE: int = 100
TARGET_LOSS: float = 0.01


class DynamicActivationNeuron(nn.Module):
    """
    Dynamic Activation Neurons (DANs) class.
    This class defines a universal neural network module that can apply various transformations and activation functions.
    The output can be scaled by a learnable parameter.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        scale_output: bool = True,
        activation_functions: Optional[
            List[Callable[[torch.Tensor], torch.Tensor]]
        ] = None,
        neuron_types: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    ):
        """
        Initialize the DynamicActivationNeuron module.

        Parameters:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        scale_output (bool): Whether to scale the output by a learnable parameter.
        activation_functions (Optional[List[Callable[[torch.Tensor], torch.Tensor]]]): List of activation functions to use.
        neuron_types (Optional[List[Callable[[torch.Tensor], torch.Tensor]]]): List of neuron type specific processing functions to use.
        """
        super().__init__()

        # Define the basis function as a linear transformation
        self.basis_function = nn.Linear(input_size, output_size)

        # Define the learnable scaling parameter if scale_output is True
        self.param = (
            nn.Parameter(torch.randn(output_size, dtype=torch.float))
            if scale_output
            else None
        )

        # Define the default activation functions based on the Kolmogorov-Arnold representation theorem
        self.activation_functions = activation_functions or [
            F.relu,
            torch.sigmoid,
            torch.tanh,
            self.spiking_activation,
            self.polynomial_activation,
            self.fourier_activation,
        ]

        # Define the default neuron types based on the Kolmogorov-Arnold representation theorem
        self.neuron_types = neuron_types or [
            self.spiking_neuron,
            self.graph_neuron,
            self.identity_neuron,
            self.polynomial_neuron,
            self.fourier_neuron,
        ]

        # Define learnable weights for the activation functions and neuron types
        self.activation_weights = nn.Parameter(
            torch.randn(len(self.activation_functions), dtype=torch.float)
        )
        self.neuron_type_weights = nn.Parameter(
            torch.randn(len(self.neuron_types), dtype=torch.float)
        )

        logger.info(
            f"DynamicActivationNeuron initialized with input_size={input_size}, output_size={output_size}, scale_output={scale_output}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DynamicActivationNeuron module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the linear transformation, activation function, and scaling.
        """
        try:
            # Ensure input is float
            x = x.float()
            logger.debug(f"Input tensor converted to float: {x}")

            # Apply the basis function (linear transformation)
            output = self.basis_function(x)

            # Apply the learned activation function
            output = self.apply_learned_activation(output)

            # Apply the learned neuron type specific processing
            output = self.apply_learned_neuron_type(output)

            # Scale the output if scale_output is True
            if self.param is not None:
                output = self.param * output

            logger.debug(f"Output tensor after transformation and activation: {output}")
            return output
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            raise

    def apply_learned_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a learned combination of activation functions.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the activation functions.
        """
        # Compute the weighted sum of activation functions
        activation_output = sum(
            weight * activation(x)
            for weight, activation in zip(
                self.activation_weights, self.activation_functions
            )
        )
        return activation_output

    def apply_learned_neuron_type(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a learned combination of neuron type specific processing.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Processed tensor.
        """
        # Compute the weighted sum of neuron type specific processing
        neuron_type_output = sum(
            weight * neuron(x)
            for weight, neuron in zip(self.neuron_type_weights, self.neuron_types)
        )
        return neuron_type_output

    def spiking_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a spiking activation function.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the spiking activation.
        """
        return (x > 0).float()

    def polynomial_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a polynomial activation function.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the polynomial activation.
        """
        return x**2

    def fourier_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a Fourier activation function.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the Fourier activation.
        """
        return torch.sin(x)

    def spiking_neuron(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the tensor as a spiking neuron.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Processed tensor.
        """
        return x * torch.exp(-x)

    def graph_neuron(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the tensor as a graph neuron.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Processed tensor.
        """
        return torch.mean(x, dim=0, keepdim=True)

    def identity_neuron(self, x: torch.Tensor) -> torch.Tensor:
        """
        Identity function for neuron processing.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor, unchanged.
        """
        return x

    def polynomial_neuron(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the tensor as a polynomial neuron.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Processed tensor.
        """
        return x**3

    def fourier_neuron(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the tensor as a Fourier neuron.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Processed tensor.
        """
        return torch.cos(x)

    def extra_repr(self) -> str:
        """
        Extra representation of the module for better debugging and logging.
        """
        return f"input_size={self.basis_function.in_features}, output_size={self.basis_function.out_features}, scale_output={self.param is not None}"

    def __getstate__(self):
        """
        Get the state of the object for pickling.
        """
        state = self.__dict__.copy()
        state["activation_functions"] = [f.__name__ for f in self.activation_functions]
        state["neuron_types"] = [f.__name__ for f in self.neuron_types]
        return state

    def __setstate__(self, state):
        """
        Set the state of the object for unpickling.
        """
        self.__dict__.update(state)
        self.activation_functions = [
            getattr(F, name) if hasattr(F, name) else getattr(self, name)
            for name in state["activation_functions"]
        ]
        self.neuron_types = [getattr(self, name) for name in state["neuron_types"]]


def main():
    """
    Main function to test and train the DynamicActivationNeuron module.
    """
    # Define constants
    MD_DIRECTORY: str = "/home/lloyd/Downloads/gpt_chats"
    PROCESSED_DATA_DIR: str = "/media/lloyd/Aurora_M2/dandata/processed_data/"
    SYNTHETIC_DATA_DIR: str = "/media/lloyd/Aurora_M2/dandata/synthetic_data/"
    UNTRAINED_MODEL_PATH: str = "/media/lloyd/Aurora_M2/dandata/untrained_dan_model.pkl"
    REGRESSION_TRAINED_MODEL_PATH: str = (
        "/media/lloyd/Aurora_M2/dandata/regression_trained_dan_model.pkl"
    )
    TEXT_TRAINED_MODEL_PATH: str = (
        "/media/lloyd/Aurora_M2/dandata/text_trained_dan_model.pkl"
    )
    OUTPUT_DIR: str = "/media/lloyd/Aurora_M2/dandata/outputs/"
    INPUT_SIZE: int = 1024
    OUTPUT_SIZE: int = 1  # Set target to 1 to fix tensor size mismatch
    SYNTHETIC_DATA_SIZE: int = 1000000
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    LR: float = 0.001
    WEIGHT_DECAY: float = 0.0001
    SCALE_OUTPUT: bool = True
    VECTORIZER: TfidfVectorizer = TfidfVectorizer(max_features=INPUT_SIZE)
    BATCH_SIZE: int = 100
    TARGET_LOSS: float = 0.01

    def ensure_directories_exist():
        """Ensure all necessary directories exist."""
        paths = [
            UNTRAINED_MODEL_PATH,
            REGRESSION_TRAINED_MODEL_PATH,
            TEXT_TRAINED_MODEL_PATH,
            SYNTHETIC_DATA_DIR,
            PROCESSED_DATA_DIR,
            OUTPUT_DIR,
        ]
        for path in paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def load_or_create_model():
        """Open a file browser to select an untrained/trained model or create a new one."""
        from tkinter import Tk, filedialog

        # Hide the root window
        root = Tk()
        root.withdraw()

        # Open file dialog
        model_path = filedialog.askopenfilename(
            initialdir=os.path.dirname(UNTRAINED_MODEL_PATH),
            title="Select a Model File",
            filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*")),
        )

        if model_path:
            model = torch.load(model_path)
            logger.info(f"Loaded model from {model_path}.")
        else:
            model = DynamicActivationNeuron(INPUT_SIZE, OUTPUT_SIZE, SCALE_OUTPUT)
            torch.save(model, UNTRAINED_MODEL_PATH)
            logger.info("Created and saved a new untrained model.")

        return model

    def process_and_prepare_md_data(
        md_directory: str,
        vectorizer: TfidfVectorizer,
        input_size: int,
        output_size: int,
        test_size: float,
        random_state: int,
        batch_size: int,
    ) -> Tuple[DataLoader, DataLoader, int, int]:
        """
        Process and prepare markdown data for training and validation.

        Parameters:
        md_directory (str): Directory containing markdown files.
        vectorizer (TfidfVectorizer): Vectorizer for text data.
        input_size (int): Input size for the model.
        output_size (int): Output size for the model.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.
        batch_size (int): Batch size for data loaders.

        Returns:
        Tuple[DataLoader, DataLoader, int, int]: Training and validation data loaders, input size, and output size.
        """

        def parse_md_files(directory: str) -> List[str]:
            """
            Parse all markdown files in the given directory and its subdirectories.

            Parameters:
            directory (str): The root directory to search for markdown files.

            Returns:
            List[str]: List of parsed text content from markdown files.
            """
            texts = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".md"):
                        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                            texts.append(f.read())
            return texts

        def extract_labels(texts: List[str]) -> List[int]:
            """
            Extract labels from the parsed markdown texts.

            Parameters:
            texts (List[str]): List of parsed text content from markdown files.

            Returns:
            List[int]: List of labels extracted from the texts.
            """
            labels = []
            for text in texts:
                if "## USER" in text:
                    labels.append(0)
                elif "## ASSISTANT" in text:
                    labels.append(1)
                else:
                    labels.append(-1)  # Unknown label
            return labels

        texts = []
        for subdir in os.listdir(md_directory):
            subdir_path = os.path.join(md_directory, subdir)
            if os.path.isdir(subdir_path):
                texts.extend(parse_md_files(subdir_path))
        labels = extract_labels(texts)

        # Filter out texts with unknown labels
        texts, labels = zip(
            *[(text, label) for text, label in zip(texts, labels) if label != -1]
        )

        # Ensure the vectorizer is set up with the correct input size
        vectorizer.max_features = input_size
        X = vectorizer.fit_transform(texts).toarray()

        # Check if the number of features matches the input size
        if X.shape[1] != input_size:
            raise ValueError(
                f"Feature size of the vectorized texts {X.shape[1]} does not match the expected input size {input_size}."
            )

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, labels, test_size=test_size, random_state=random_state
        )

        # Ensure the output size is correct
        unique_labels = np.unique(labels)
        if len(unique_labels) > output_size:
            raise ValueError(
                f"Number of unique labels {len(unique_labels)} exceeds the output size {output_size}."
            )

        # Save processed data
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        np.savetxt(
            os.path.join(PROCESSED_DATA_DIR, "combined_dataset.txt"), X, fmt="%f"
        )
        np.savetxt(
            os.path.join(PROCESSED_DATA_DIR, "train_dataset.txt"), X_train, fmt="%f"
        )
        np.savetxt(os.path.join(PROCESSED_DATA_DIR, "val_dataset.txt"), X_val, fmt="%f")

        # Create DataLoader for training and validation sets
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, input_size, output_size

    def create_or_load_synthetic_data(
        input_size: int = INPUT_SIZE,
        output_size: int = OUTPUT_SIZE,
        n_samples: int = SYNTHETIC_DATA_SIZE,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create synthetic dataset if it doesn't exist, otherwise load it.

        Parameters:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        n_samples (int): Number of samples in the dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.

        Returns:
        Tuple[DataLoader, DataLoader]: Train and validation data loaders.
        """
        if not os.path.exists(os.path.join(SYNTHETIC_DATA_DIR, "synthetic_data.npy")):
            X, y = make_regression(
                n_samples=n_samples,
                n_features=input_size,
                n_targets=output_size,
                noise=0.1,
            )
            if output_size == 1:
                y = y.reshape(-1, 1)
            np.save(os.path.join(SYNTHETIC_DATA_DIR, "synthetic_data.npy"), X)
            np.save(os.path.join(SYNTHETIC_DATA_DIR, "synthetic_labels.npy"), y)
            logger.info("Created and saved synthetic dataset.")
        else:
            X = np.load(os.path.join(SYNTHETIC_DATA_DIR, "synthetic_data.npy"))
            y = np.load(os.path.join(SYNTHETIC_DATA_DIR, "synthetic_labels.npy"))
            logger.info("Loaded existing synthetic dataset.")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        return train_loader, val_loader

    def train_and_save_model(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        path: str,
        target_loss: float = TARGET_LOSS,
        max_epochs: int = 1000,
        save_interval: int = 10,
        overfit_thresholds: list = [(100, 2), (50, 4), (25, 6), (20, 8), (10, 10)],
        noise_std: float = 0.01,
        lr_decay: float = 0.9,
        weight_decay_increase: float = 1.1,
    ) -> dict:
        """
        Train the model and save it to the specified path.

        Parameters:
        model (nn.Module): The neural network model to train.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        path (str): The path to save the trained model.
        target_loss (float): The target loss to stop training.
        max_epochs (int): Maximum number of epochs to train.
        save_interval (int): Interval of epochs to save the model.
        overfit_thresholds (list): List of tuples for overfitting thresholds.
        noise_std (float): Standard deviation of noise to add for overfitting.
        lr_decay (float): Learning rate decay factor.
        weight_decay_increase (float): Weight decay increase factor.

        Returns:
        dict: Dictionary containing training and validation loss history.
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_rmse": [],
            "val_rmse": [],
            "train_mape": [],
            "val_mape": [],
        }

        epoch = 0
        overfit_count = 0
        initial_lr = optimizer.param_groups[0]["lr"]
        initial_weight_decay = optimizer.param_groups[0]["weight_decay"]
        stop_training = False

        def calculate_metrics(outputs, labels):
            loss = criterion(outputs, labels)
            rmse = torch.sqrt(torch.mean((outputs - labels) ** 2)).item()
            mape = torch.mean(torch.abs((labels - outputs) / labels)).item()
            return loss, rmse, mape

        def get_overfit_threshold(val_loss):
            for threshold, epochs in overfit_thresholds:
                if val_loss > threshold:
                    return epochs
            return 10

        def user_wants_to_stop() -> bool:
            """
            Check if the user wants to stop the training process.
            This function checks for a specific interrupt signal (SIGUSR1) sent by the user.
            """
            import signal

            stop_signal_received = False

            def signal_handler(sig, frame):
                nonlocal stop_signal_received
                stop_signal_received = True

            prev_handler = signal.signal(signal.SIGUSR1, signal_handler)

            if stop_signal_received:
                print("Stop signal received. Finalizing training...")
                signal.signal(signal.SIGUSR1, prev_handler)
                return True
            else:
                signal.signal(signal.SIGUSR1, prev_handler)
                return False

        while epoch < max_epochs and not stop_training:
            model.train()
            running_train_loss = 0.0
            running_train_rmse = 0.0
            running_train_mape = 0.0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)

                if outputs.shape != labels.shape:
                    outputs = outputs.view(-1)
                    labels = labels.view(-1)

                loss, rmse, mape = calculate_metrics(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * inputs.size(0)
                running_train_rmse += rmse * inputs.size(0)
                running_train_mape += mape * inputs.size(0)

            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            epoch_train_rmse = running_train_rmse / len(train_loader.dataset)
            epoch_train_mape = running_train_mape / len(train_loader.dataset)
            history["train_loss"].append(epoch_train_loss)
            history["train_rmse"].append(epoch_train_rmse)
            history["train_mape"].append(epoch_train_mape)

            model.eval()
            running_val_loss = 0.0
            running_val_rmse = 0.0
            running_val_mape = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)

                    if outputs.shape != labels.shape:
                        outputs = outputs.view(-1)
                        labels = labels.view(-1)

                    loss, rmse, mape = calculate_metrics(outputs, labels)
                    running_val_loss += loss.item() * inputs.size(0)
                    running_val_rmse += rmse * inputs.size(0)
                    running_val_mape += mape * inputs.size(0)

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            epoch_val_rmse = running_val_rmse / len(val_loader.dataset)
            epoch_val_mape = running_val_mape / len(val_loader.dataset)
            history["val_loss"].append(epoch_val_loss)
            history["val_rmse"].append(epoch_val_rmse)
            history["val_mape"].append(epoch_val_mape)

            logger.info(
                f"Epoch {epoch + 1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train RMSE: {epoch_train_rmse:.4f}, Val RMSE: {epoch_val_rmse:.4f}, Train MAPE: {epoch_train_mape:.4f}, Val MAPE: {epoch_val_mape:.4f}"
            )

            if (epoch + 1) % save_interval == 0:
                torch.save(model.state_dict(), path)
                logger.info(f"Saved the model state to {path} at epoch {epoch + 1}.")
                plot_metrics(history)
                clear_memory()

            if epoch_val_loss <= target_loss:
                logger.info(f"Target loss {target_loss} reached at epoch {epoch + 1}.")
                break

            if epoch > 0:
                current_val_loss = history["val_loss"][-1]
                previous_val_loss = history["val_loss"][-2]
                overfit_epochs_required = get_overfit_threshold(current_val_loss)

                if current_val_loss > previous_val_loss:
                    overfit_count += 1
                else:
                    overfit_count = 0

                if overfit_count >= overfit_epochs_required:
                    logger.info("Overfitting detected, adding noise to parameters.")
                    with torch.no_grad():
                        for param in model.parameters():
                            noise = torch.normal(
                                mean=0.0, std=noise_std, size=param.size()
                            )
                            param.add_(noise)
                    overfit_count = 0

                    # Adjust learning rate and weight decay dynamically
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = max(
                            param_group["lr"] * lr_decay, initial_lr * 0.1
                        )
                        param_group["weight_decay"] = min(
                            param_group["weight_decay"] * weight_decay_increase,
                            initial_weight_decay * 10,
                        )

            epoch += 1

            # Check for user interruption to stop training
            if user_wants_to_stop():
                logger.info("Training stopped by user.")
                stop_training = True

        if not stop_training:
            torch.save(model.state_dict(), path)
            logger.info(f"Saved the final model state to {path}.")
        return history

    def predict(model: nn.Module, data_loader: DataLoader) -> torch.Tensor:
        """
        Predict function for the model.

        Parameters:
        model (nn.Module): The neural network model.
        data_loader (DataLoader): The data loader.

        Returns:
        torch.Tensor: Predictions.
        """
        model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                outputs = model(inputs)
                predictions.append(outputs)
        return torch.cat(predictions, dim=0)

    def log_predictions(predictions: torch.Tensor) -> None:
        """
        Log the mean and standard deviation of the predictions.

        Parameters:
        predictions (torch.Tensor): The predictions tensor.
        """
        try:
            mean_prediction = torch.mean(predictions, dim=0)
            std_prediction = torch.std(predictions, dim=0)
            logger.info(f"Mean of predictions: {mean_prediction}")
            logger.info(f"Standard deviation of predictions: {std_prediction}")
        except Exception as e:
            logger.error(f"Error during prediction analysis: {e}")

    def plot_metrics(history: dict) -> None:
        """
        Plot the training and validation loss, RMSE, and MAPE.

        Parameters:
        history (dict): Dictionary containing training and validation metrics history.
        """
        epochs = range(len(history["train_loss"]))
        plt.figure(figsize=(18, 10))

        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, history["train_loss"], label="Training Loss")
        plt.plot(epochs, history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"Training and Validation Loss Over Time (Best Val Loss: {min(history['val_loss']):.4f})"
        )
        plt.legend()

        # Plot RMSE
        plt.subplot(2, 2, 2)
        plt.plot(epochs, history["train_rmse"], label="Training RMSE")
        plt.plot(epochs, history["val_rmse"], label="Validation RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title(
            f"Training and Validation RMSE Over Time (Best Val RMSE: {min(history['val_rmse']):.4f})"
        )
        plt.legend()

        # Plot MAPE
        plt.subplot(2, 2, 3)
        plt.plot(epochs, history["train_mape"], label="Training MAPE")
        plt.plot(epochs, history["val_mape"], label="Validation MAPE")
        plt.xlabel("Epoch")
        plt.ylabel("MAPE")
        plt.title(
            f"Training and Validation MAPE Over Time (Best Val MAPE: {min(history['val_mape']):.4f})"
        )
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(OUTPUT_DIR, f"metrics_epoch_{len(epochs)}.png"))
        plt.close()

    def clear_memory():
        """Clear RAM and ensure that the model is the only thing in memory.
        This program focuses on CPU usage mainly but CUDA cache cleared also.
        """
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    try:
        ensure_directories_exist()
        dan = load_or_create_model()
        regression_train_loader, regression_val_loader = create_or_load_synthetic_data()

        # Train the model on synthetic data
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(dan.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        history = train_and_save_model(
            dan,
            criterion,
            optimizer,
            regression_train_loader,
            regression_val_loader,
            REGRESSION_TRAINED_MODEL_PATH,
        )
        plot_metrics(history)
        clear_memory()

        # Predict using the trained model on regression data
        predictions = predict(dan, regression_val_loader)
        log_predictions(predictions)

        text_train_loader, text_val_loader, _, _ = process_and_prepare_md_data(
            MD_DIRECTORY,
            VECTORIZER,
            INPUT_SIZE,
            OUTPUT_SIZE,
            TEST_SIZE,
            RANDOM_STATE,
            BATCH_SIZE,
        )
        # Train the model with text data
        criterion = nn.CrossEntropyLoss()
        history = train_and_save_model(
            dan,
            criterion,
            optimizer,
            text_train_loader,
            text_val_loader,
            TEXT_TRAINED_MODEL_PATH,
        )
        plot_metrics(history)
        clear_memory()

        # Predict using the trained model on text data
        predictions = predict(dan, text_val_loader)
        log_predictions(predictions)

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
