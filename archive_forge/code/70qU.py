import logging
import random
from typing import Optional, List, Dict, Any, Tuple
from tkinter import filedialog, messagebox
import tkinter as tk
import torch
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from kan import KAN

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class KANWrapper:
    """
    A wrapper class for the Kolmogorov-Arnold Network (KAN).

    Attributes:
        width (List[int]): List of integers specifying the width of each layer.
        grid (int): Grid size for the KAN.
        k (int): Number of neurons in the hidden layer.
        learnable_params (Dict[str, Any]): Dictionary of additional learnable parameters.
        model (KAN): The KAN model instance.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
    """

    def __init__(
        self,
        width: List[int],
        grid: int,
        k: int,
        learnable_params: Optional[Dict[str, Any]] = None,
        lr: float = 0.01,
    ) -> None:
        """
        Initialize the Kolmogorov-Arnold Network (KAN).

        Args:
            width (List[int]): List of integers specifying the width of each layer.
            grid (int): Grid size for the KAN.
            k (int): Number of neurons in the hidden layer.
            learnable_params (Optional[Dict[str, Any]]): Optional dictionary of additional learnable parameters.
            lr (float): Learning rate for the optimizer.
        """
        self.width = width
        self.grid = grid
        self.k = k
        self.learnable_params = learnable_params or {}
        self.model = KAN(width=width, grid=grid, k=k)
        self._initialize_parameters()
        self._initialize_missing_keys()
        self.optimizer = Adam(
            self.model.parameters(), lr=self.learnable_params.get("lr", lr)
        )
        logging.info(
            "Initialized KANWrapper with width: %s, grid: %d, k: %d", width, grid, k
        )

    def _initialize_parameters(self) -> None:
        """
        Initialize all model parameters and learnable parameters with log-normal distribution.
        """
        for param in self.model.parameters():
            self._initialize_tensor(param)

        for key, value in self.learnable_params.items():
            self.learnable_params[key] = self._initialize_tensor(
                value if isinstance(value, torch.Tensor) else torch.tensor(value)
            )

    @staticmethod
    def _initialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        Initialize a tensor with log-normal distribution.

        Args:
            tensor (torch.Tensor): The tensor to initialize.

        Returns:
            torch.Tensor: The initialized tensor.
        """
        return torch.randn_like(tensor).log_normal_()

    def _initialize_missing_keys(self) -> None:
        """
        Ensure all required keys are present in learnable_params and initialize them if missing.
        """
        required_keys = self._get_required_keys()

        for key in required_keys:
            if key not in self.learnable_params:
                self.learnable_params[key] = self._initialize_tensor(torch.randn(1))
                logging.debug(
                    "Initialized missing key %s with log-normal distribution", key
                )

    def _get_required_keys(self) -> List[str]:
        """
        Get the list of required keys for learnable parameters dynamically from the model.

        Returns:
            List[str]: List of required keys.
        """
        return [name for name, _ in self.model.named_parameters()]

    def save(self, filename: str) -> None:
        """
        Save the network's weights, biases, and learnable parameters to a file.

        Args:
            filename (str): The file path to save the network.
        """
        state = {
            "model_state_dict": self.model.state_dict(),
            "width": self.width,
            "grid": self.grid,
            "k": self.k,
            "learnable_params": self.learnable_params,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        with open(filename, "wb") as f:
            torch.save(state, f)
        logging.info("Saved KANWrapper model to %s", filename)

    @classmethod
    def load(cls, filename: str) -> "KANWrapper":
        """
        Load a network's weights, biases, and learnable parameters from a file.

        Args:
            filename (str): The file path to load the network from.

        Returns:
            KANWrapper: An instance of the KANWrapper class with loaded weights, biases, and learnable parameters.
        """
        with open(filename, "rb") as file:
            state = torch.load(file)

        # Define required keys with default log-normal initialized values
        required_keys = {
            "width": torch.randn(1).log_normal_(),
            "grid": torch.randn(1).log_normal_(),
            "k": torch.randn(1).log_normal_(),
            "model_state_dict": {},
            "learnable_params": {},
            "optimizer_state_dict": {},
        }

        # Identify and fill missing keys
        missing_keys = [key for key in required_keys if key not in state]
        for key in missing_keys:
            state[key] = required_keys[key]

        if missing_keys:
            logging.warning(
                "Missing keys %s were initialized with default log-normal values",
                missing_keys,
            )

        # Initialize the KANWrapper instance with the loaded state
        instance = cls(
            width=state["width"],
            grid=state["grid"],
            k=state["k"],
            learnable_params=state["learnable_params"],
        )
        instance.model.load_state_dict(state["model_state_dict"])
        instance.optimizer.load_state_dict(state["optimizer_state_dict"])

        return instance

    def initialize_from_another_model(
        self, other: "KANWrapper", input_tensor: torch.Tensor
    ) -> None:
        """
        Initialize the model from another parent model.

        Args:
            other (KANWrapper): The parent model to initialize from.
            input_tensor (torch.Tensor): Input tensor for initialization.
        """
        try:
            # Validate input types
            if not isinstance(other, KANWrapper):
                raise TypeError("other must be an instance of KANWrapper")
            if not isinstance(input_tensor, torch.Tensor):
                raise TypeError("input_tensor must be a torch.Tensor")

            # Initialize model from another model
            self.model.initialize_from_another_model(other.model, input_tensor)
            logging.info("Initialized KANWrapper from another model")

            # Dynamically determine and initialize missing parameters
            self._initialize_missing_keys()

            # Save the initialized model state
            if messagebox.askyesno(
                "Save Initialized Model",
                "Model initialized from another model. Do you want to save the initialized model?",
            ):
                filename = filedialog.asksaveasfilename(
                    defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")]
                )
                if filename:
                    self.save(filename)
                    logging.info("Saved initialized model to %s", filename)

        except (TypeError, ValueError) as e:
            logging.error(
                "TypeError or ValueError in initialize_from_another_model: %s", e
            )
            raise
        except RuntimeError as e:
            logging.critical("RuntimeError in initialize_from_another_model: %s", e)
            raise
        except Exception as e:
            logging.error("Unexpected error in initialize_from_another_model: %s", e)
            raise

    def fix_symbolic_function(
        self, l: int, i: int, j: int, expression: str, fit_parameters: bool = True
    ) -> None:
        """Fix a symbolic function in the KAN model."""
        try:
            self.model.fix_symbolic(l, i, j, expression, fit_parameters)
        except AttributeError as e:
            logging.error("Unexpected error in fix_symbolic_function: %s", e)
            raise

    def get_activation_range(
        self, l: int, i: int, j: int
    ) -> Tuple[float, float, float, float]:
        """Get the activation range for a specific neuron."""
        return self.model.get_range(l, i, j)

    def plot_results(
        self, x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor
    ) -> None:
        """Plot the results of the model's predictions."""
        plt.figure(figsize=(10, 5))
        plt.scatter(x[:, 0].numpy(), y[:, 0].numpy(), label="True Values")
        plt.scatter(
            x[:, 0].numpy(), y_pred[:, 0].detach().numpy(), label="Predicted Values"
        )
        plt.legend()
        plt.show()
        logging.info("Plotted results")

    def train(
        self,
        dataset: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 0.01,
        epochs: int = 100,
    ) -> None:
        """
        Train the network using a simple gradient descent algorithm.

        Args:
            dataset (torch.Tensor): Input data for training.
            targets (torch.Tensor): Target outputs for training.
            learning_rate (float): Learning rate for gradient descent.
            epochs (int): Number of training epochs.
        """
        optimizer = SGD(self.model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            for x, y in zip(dataset, targets):
                optimizer.zero_grad()
                output = self.forward(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def mutate(self, mutation_rate: float = 0.1) -> None:
        """
        Mutate the network's weights and biases.

        Args:
            mutation_rate (float): Probability of mutation for each weight and bias.
        """
        for param in self.model.parameters():
            if torch.rand(1).item() < mutation_rate:
                param.data += torch.randn_like(param) * mutation_rate
                logging.debug(f"Mutated parameter with shape {param.shape}")

    def inherit(self, other: "KANWrapper") -> None:
        """
        Inherit weights and biases from another KAN instance.

        Args:
            other (KANWrapper): Another KAN instance to inherit from.
        """
        self.model.load_state_dict(other.model.state_dict())
        logging.info("Inherited model parameters from another instance.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        return self.model(x)


if __name__ == "__main__":
    # Example usage and comprehensive testing of KANWrapper functionalities

    # Initialize KANWrapper
    width = [2, 5, 3]
    grid = 5
    k = 3
    learnable_params = {"lr": 0.01}
    kan_wrapper = KANWrapper(width, grid, k, learnable_params)

    # Save the model
    kan_wrapper.save("kan_model.pth")

    # Load the model
    loaded_kan_wrapper = KANWrapper.load("kan_model.pth")

    # Initialize from another model
    input_tensor = torch.randn(1, 2)
    kan_wrapper.initialize_from_another_model(loaded_kan_wrapper, input_tensor)

    # Get activation range
    activation_range = kan_wrapper.get_activation_range(0, 0, 0)
    print("Activation Range:", activation_range)

    # Create dummy data for training
    dataset = torch.normal(0, 1, size=(100, 2))
    targets = torch.normal(0, 1, size=(100, 3))

    # Train the KANWrapper
    kan_wrapper.train(dataset, targets, learning_rate=0.01, epochs=10)

    # Mutate the model
    kan_wrapper.mutate(mutation_rate=0.1)

    # Inherit from another model
    kan_wrapper.inherit(loaded_kan_wrapper)

    # Perform a forward pass with the loaded model
    test_input = torch.normal(0, 1, size=(1, 2))
    output = loaded_kan_wrapper.forward(test_input)
    print(f"Output for test input: {output}")

    # Plot results
    y_pred = kan_wrapper.forward(dataset)
    kan_wrapper.plot_results(dataset, targets, y_pred)
