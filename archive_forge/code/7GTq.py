"""
KANWrapper.py

This module provides the KANWrapper class, which encapsulates the Kolmogorov-Arnold Network (KAN) 
with additional functionalities for initialization, training, mutation, inheritance, and symbolic function fixing.

Dependencies:
- logging
- random
- typing
- tkinter
- torch
- matplotlib

Usage:
    from KANWrapper import KANWrapper
"""

from datetime import datetime
import logging
import random
from typing import Optional, List, Dict, Any, Tuple
from tkinter import filedialog, messagebox
import tkinter as tk
import torch
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from kan import KAN  # Ensure this is the correct import from the pykan module
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import TensorDataset, DataLoader


# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EnhancedKAN(KAN):
    def __init__(self, width: List[int], grid: int, k: int):
        super().__init__(width=width, grid=grid, k=k)
        self.acts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.acts = super().forward(x)
        return self.acts


class KANWrapper(EnhancedKAN):
    """
    A wrapper class for the Kolmogorov-Arnold Network (KAN).

    Attributes:
        width (List[int]): List of integers specifying the width of each layer.
        grid (int): Grid size for the KAN.
        k (int): Number of neurons in the hidden layer.
        learnable_params (Dict[str, Any]): Dictionary of additional learnable parameters.
        model (EnhancedKAN): The KAN model instance.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        epochs (int): Number of training epochs.
    """

    def __init__(
        self,
        width: List[int],
        grid: int,
        k: int,
        learnable_params: Optional[Dict[str, Any]] = None,
        lr: float = 0.01,
        epochs: int = 100,
    ) -> None:
        """
        Initialize the Kolmogorov-Arnold Network (KAN).

        Args:
            width (List[int]): List of integers specifying the width of each layer.
            grid (int): Grid size for the KAN.
            k (int): Number of neurons in the hidden layer.
            learnable_params (Optional[Dict[str, Any]]): Optional dictionary of additional learnable parameters.
            lr (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
        """
        try:
            # Initialize attributes
            self.width = width
            self.grid = grid
            self.k = k
            self.epochs = epochs
            self.learnable_params = learnable_params or {}

            # Log the initialization parameters
            logging.debug(
                "Initializing KANWrapper with width: %s, grid: %d, k: %d, learnable_params: %s, lr: %f, epochs: %d",
                width,
                grid,
                k,
                self.learnable_params,
                lr,
                epochs,
            )

            # Initialize the EnhancedKAN model
            self.model = EnhancedKAN(width=width, grid=grid, k=k)

            # Initialize parameters and missing keys
            self._initialize_parameters()
            self._initialize_missing_keys()

            # Set up the optimizer
            self.optimizer = SGD(
                self.model.parameters(), lr=self.learnable_params.get("lr", lr)
            )

            # Log successful initialization
            logging.info(
                "Successfully initialized KANWrapper with width: %s, grid: %d, k: %d, learnable_params: %s, lr: %f, epochs: %d",
                width,
                grid,
                k,
                self.learnable_params,
                lr,
                epochs,
            )
        except Exception as e:
            logging.critical("Failed to initialize KANWrapper: %s", e)
            raise

    def _initialize_parameters(self) -> None:
        """
        Initialize all model parameters and learnable parameters with log-normal distribution.
        """
        try:
            for param in self.model.parameters():
                self._initialize_tensor(param)

            for key, value in self.learnable_params.items():
                self.learnable_params[key] = self._initialize_tensor(
                    value if isinstance(value, torch.Tensor) else torch.tensor(value)
                )
            logging.debug("Initialized all model parameters and learnable parameters.")
        except Exception as e:
            logging.error("Error initializing parameters: %s", e)
            raise

    @staticmethod
    def _initialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        Initialize a tensor with log-normal distribution.

        Args:
            tensor (torch.Tensor): The tensor to initialize.

        Returns:
            torch.Tensor: The initialized tensor.
        """
        try:
            initialized_tensor = torch.randn_like(tensor).log_normal_()
            logging.debug("Initialized tensor with shape: %s", tensor.shape)
            return initialized_tensor
        except Exception as e:
            logging.error("Error initializing tensor: %s", e)
            raise

    def _initialize_missing_keys(self) -> None:
        """
        Ensure all required keys are present in learnable_params and initialize them if missing.
        """
        try:
            required_keys = self._get_required_keys()

            for key in required_keys:
                if key not in self.learnable_params:
                    self.learnable_params[key] = self._initialize_tensor(torch.randn(1))
                    logging.debug(
                        "Initialized missing key %s with log-normal distribution", key
                    )
            logging.debug(
                "Checked and initialized all missing keys in learnable_params."
            )
        except Exception as e:
            logging.error("Error initializing missing keys: %s", e)
            raise

    def _get_required_keys(self) -> List[str]:
        """
        Get the list of required keys for learnable parameters dynamically from the model.

        Returns:
            List[str]: List of required keys.
        """
        try:
            required_keys = [name for name, _ in self.model.named_parameters()]
            logging.debug(
                "Retrieved required keys for learnable parameters: %s", required_keys
            )
            return required_keys
        except Exception as e:
            logging.error("Error retrieving required keys: %s", e)
            raise

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
        with open(filename, "wb+") as f:
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

            # Ensure the input tensor has the correct dimensions
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)

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
        data_loader: DataLoader,
        val_data: TensorDataset,
        val_targets: TensorDataset,
        dataset_name: str = "default_dataset",
    ) -> None:
        """
        Train the network using a simple gradient descent algorithm with validation.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_data (torch.Tensor): Validation input data.
            val_targets (TensorDataset): Validation target outputs.
            epochs (int): Number of training epochs.
            dataset_name (str): Name of the dataset for saving model checkpoints.
        """
        epochs = 10
        lr = self.learnable_params.get("lr", 0.01)
        best_rmse = float("inf")
        overfitting_count = 0
        optimizer = SGD(self.model.parameters(), lr=self.learnable_params.get("lr", lr))
        loss_fn = torch.nn.MSELoss()

        # Ensure input tensors have the correct dimensions
        if val_data.dim() == 1:
            val_data = val_data.unsqueeze(0)
        if val_targets.dim() == 1:
            val_targets = val_targets.unsqueeze(0)
        logging.debug(
            f"Validation Data shape: {val_data.shape}, Validation Targets shape: {val_targets.shape}"
        )

        for epoch in range(self.epochs):
            self.model.train()
            for batch_data, batch_targets in data_loader:
                if batch_data.dim() == 1:
                    batch_data = batch_data.unsqueeze(0)
                if batch_targets.dim() == 1:
                    batch_targets = batch_targets.unsqueeze(0)
                logging.debug(
                    f"Batch Data shape: {batch_data.shape}, Batch Targets shape: {batch_targets.shape}"
                )

                optimizer.zero_grad()
                output = self.forward(batch_data)
                loss = loss_fn(output, batch_targets)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                if val_data.dim() == 1:
                    val_data = val_data.unsqueeze(0)
                if val_targets.dim() == 1:
                    val_targets = val_targets.unsqueeze(0)
                y_pred = self.forward(val_data)
                validation_loss = loss_fn(y_pred, val_targets).item()
                rmse = torch.sqrt(torch.mean((y_pred - val_targets) ** 2)).item()
                mape = (
                    torch.mean(torch.abs((y_pred - val_targets) / val_targets)).item()
                    * 100
                )
                variance = torch.var(y_pred).item()
                std_dev = torch.std(y_pred).item()

            logging.info(
                f"Epoch {epoch+1}/{epochs}, Validation Loss: {validation_loss:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%, Variance: {variance:.4f}, Std Dev: {std_dev:.4f}"
            )

            # Save the model periodically
            if (epoch + 1) % 50 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"outputs/{dataset_name}"
                os.makedirs(output_dir, exist_ok=True)
                self.save(f"{output_dir}/kan_model_epoch_{epoch+1}_{timestamp}.pth")

            # Check for overfitting
            if rmse < best_rmse:
                best_rmse = rmse
                overfitting_count = 0
            else:
                overfitting_count += 1
                if overfitting_count >= 3:
                    logging.info(
                        "Overfitting detected. Injecting noise and early stopping."
                    )
                    break

        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save(f"outputs/{dataset_name}/kan_model_final_{timestamp}.pth")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize KANWrapper with appropriate parameters for large multidimensional data
    width = [2, 10, 10, 3]
    grid = 10
    k = 5
    learnable_params = {"lr": 0.001}
    kan_wrapper = KANWrapper(width, grid, k, learnable_params)

    # Save the model
    kan_wrapper.save("/media/lloyd/Aurora_M2/dandata/kan_model_initial")

    # Load the model
    model = kan_wrapper.load("/media/lloyd/Aurora_M2/dandata/kan_model_initial")

    # Create a highly detailed, multidimensional dataset
    num_samples = 100000
    num_features = 10
    dataset = torch.normal(0, 1, size=(num_samples, num_features))
    targets = torch.normal(0, 1, size=(num_samples, 3))

    # Save the dataset to the most efficient format (Parquet)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = f"/media/lloyd/Aurora_M2/dandata/kan_data_{timestamp}.parquet"
    df = pd.DataFrame(dataset.numpy())
    df.to_parquet(dataset_path, index=False)
    logging.info(f"Dataset saved to {dataset_path}")

    # Clear memory
    del df

    # Split the dataset into training, validation, and testing sets
    train_data, test_data, train_targets, test_targets = train_test_split(
        dataset, targets, test_size=0.1, random_state=42
    )
    train_data, val_data, train_targets, val_targets = train_test_split(
        train_data, train_targets, test_size=0.2, random_state=42
    )

    # Save the splits
    train_path = f"/media/lloyd/Aurora_M2/dandata/kan_train_{timestamp}.parquet"
    val_path = f"/media/lloyd/Aurora_M2/dandata/kan_val_{timestamp}.parquet"
    test_path = f"/media/lloyd/Aurora_M2/dandata/kan_test_{timestamp}.parquet"
    pd.DataFrame(train_data.numpy()).to_parquet(train_path, index=False)
    pd.DataFrame(val_data.numpy()).to_parquet(val_path, index=False)
    pd.DataFrame(test_data.numpy()).to_parquet(test_path, index=False)
    logging.info(
        f"Training, validation, and testing sets saved to {train_path}, {val_path}, and {test_path}"
    )

    # Create DataLoader for batching and shuffling
    batch_size = 64
    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Ask user if they want to train on existing datasets
    train_existing = input("Train on existing datasets? (y/n): ").strip().lower() == "y"

    # Train on datasets
    if train_existing:
        logging.info("Training on existing datasets.")
        model.train(
            train_loader,
            val_data,
            val_targets,
            model,
        )
        model.train(
            val_loader,
            val_data,
            val_targets,
            model,
        )
    logging.info("Training complete.")

    # Save the trained model
    model.save(f"/media/lloyd/Aurora_M2/dandata/kan_model_trained_{timestamp}.pth")
    logging.info(
        f"Trained model saved to /media/lloyd/Aurora_M2/dandata/kan_model_trained_{timestamp}.pth"
    )

    # Perform a forward pass with the loaded model
    test_input = torch.normal(0, 1, size=(1, 2))
    logging.debug(f"Test input shape: {test_input.shape}")
    output = model.forward(test_input)
    logging.info(f"Output for test input: {output}")

    # Plot final results
    y_pred = model.forward(test_data)
    model.plot_results(test_data, test_targets, y_pred)
    logging.info("Final results plotted and saved.")
