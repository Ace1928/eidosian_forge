import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Union, Callable, List
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def train_model(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 25,
) -> dict:
    """
    Train the neural network model with advanced biological-like efficiency and robustness.

    Parameters:
    model (nn.Module): The neural network model to train.
    criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function.
    optimizer (torch.optim.Optimizer): The optimizer.
    train_loader (torch.utils.data.DataLoader): The training data loader.
    val_loader (torch.utils.data.DataLoader): The validation data loader.
    num_epochs (int): Number of epochs to train the model.

    Returns:
    dict: Dictionary containing training and validation loss history.
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_accuracy = correct_train / total_train
        history["train_loss"].append(epoch_train_loss)
        history["train_accuracy"].append(epoch_train_accuracy)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_accuracy = correct_val / total_val
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_accuracy)

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}"
        )

    return history


def predict(model: nn.Module, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Predict function for the model.

    Parameters:
    model (nn.Module): The neural network model.
    data_loader (torch.utils.data.DataLoader): The data loader.

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


def create_synthetic_dataset(
    input_size: int,
    output_size: int,
    n_samples: int = 1000000,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Create a synthetic dataset for training and validation.

    Parameters:
    input_size (int): The size of the input features.
    output_size (int): The size of the output features.
    n_samples (int): Number of samples in the dataset.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed.

    Returns:
    tuple: Train and validation data loaders.
    """

    X, y = make_regression(
        n_samples=n_samples, n_features=input_size, n_targets=output_size, noise=0.1
    )
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

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

    return train_loader, val_loader


def plot_loss(history: dict, num_epochs: int):
    """
    Plot the training and validation loss.

    Parameters:
    history (dict): Dictionary containing training and validation loss history.
    num_epochs (int): Number of epochs.
    """
    plt.figure()
    plt.plot(range(num_epochs), history["train_loss"], label="Training Loss")
    plt.plot(range(num_epochs), history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.show()


def main():
    """
    Main function to test the DynamicActivationNeuron module.
    """
    # Test initialization
    input_size = 1024
    output_size = 128
    scale_output = True
    dan = DynamicActivationNeuron(input_size, output_size, scale_output)
    logger.info("DynamicActivationNeuron instance created.")

    # Create a synthetic dataset with learnable information
    train_loader, val_loader = create_synthetic_dataset(input_size, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dan.parameters(), lr=0.0001, weight_decay=0.0001)

    # Train the model
    num_epochs = 50
    history = train_model(
        dan, criterion, optimizer, train_loader, val_loader, num_epochs
    )

    # Plot the training and validation loss
    plot_loss(history, num_epochs)

    # Test forward pass with random input
    test_input = torch.randn(2, input_size)
    logger.info(f"Test input: {test_input}")
    try:
        output = dan(test_input)
        logger.info(f"Forward pass output: {output}")
    except Exception as e:
        logger.error(f"Error during forward pass: {e}")

    # Test apply_learned_activation
    try:
        activation_output = dan.apply_learned_activation(test_input)
        logger.info(f"Activation output: {activation_output}")
    except Exception as e:
        logger.error(f"Error during apply_learned_activation: {e}")

    # Test apply_learned_neuron_type
    try:
        neuron_type_output = dan.apply_learned_neuron_type(test_input)
        logger.info(f"Neuron type output: {neuron_type_output}")
    except Exception as e:
        logger.error(f"Error during apply_learned_neuron_type: {e}")

    # Test spiking_activation
    try:
        spiking_output = dan.spiking_activation(test_input)
        logger.info(f"Spiking activation output: {spiking_output}")
    except Exception as e:
        logger.error(f"Error during spiking_activation: {e}")

    # Test spiking_neuron
    try:
        spiking_neuron_output = dan.spiking_neuron(test_input)
        logger.info(f"Spiking neuron output: {spiking_neuron_output}")
    except Exception as e:
        logger.error(f"Error during spiking_neuron: {e}")

    # Test graph_neuron
    try:
        graph_neuron_output = dan.graph_neuron(test_input)
        logger.info(f"Graph neuron output: {graph_neuron_output}")
    except Exception as e:
        logger.error(f"Error during graph_neuron: {e}")

    # Test extra_repr
    try:
        repr_output = dan.extra_repr()
        logger.info(f"Extra representation: {repr_output}")
    except Exception as e:
        logger.error(f"Error during extra_repr: {e}")

    # Test the predict function
    try:
        predictions = predict(dan, val_loader)
        logger.info(f"Predictions: {predictions}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")

    # Additional analysis: Calculate and log the mean and standard deviation of the predictions
    try:
        mean_prediction = torch.mean(predictions, dim=0)
        std_prediction = torch.std(predictions, dim=0)
        logger.info(f"Mean of predictions: {mean_prediction}")
        logger.info(f"Standard deviation of predictions: {std_prediction}")
    except Exception as e:
        logger.error(f"Error during prediction analysis: {e}")


if __name__ == "__main__":
    main()
