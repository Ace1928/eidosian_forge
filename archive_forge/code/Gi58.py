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


class KANWrapper(KAN):
    def __init__(
        self,
        width: List[int],
        grid: int = 3,
        k: int = 3,
        noise_scale: float = 0.1,
        noise_scale_base: float = 0.1,
        base_fun: torch.nn.Module = torch.nn.SiLU(),
        symbolic_enabled: bool = True,
        bias_trainable: bool = True,
        grid_eps: float = 1,
        grid_range: List[int] = [-1, 1],
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        device: str = "cpu",
        seed: int = 0,
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
            noise_scale (float): Scale of the noise.
            noise_scale_base (float): Base scale of the noise.
            base_fun (torch.nn.Module): Base function for the KAN.
            symbolic_enabled (bool): Whether symbolic functions are enabled.
            bias_trainable (bool): Whether biases are trainable.
            grid_eps (float): Epsilon for the grid.
            grid_range (List[int]): Range for the grid.
            sp_trainable (bool): Whether sp is trainable.
            sb_trainable (bool): Whether sb is trainable.
            device (str): Device to run the model on.
            seed (int): Random seed.
            learnable_params (Optional[Dict[str, Any]]): Optional dictionary of additional learnable parameters.
            lr (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
        """
        super().__init__(
            width=width,
            grid=grid,
            k=k,
            noise_scale=noise_scale,
            noise_scale_base=noise_scale_base,
            base_fun=base_fun,
            symbolic_enabled=symbolic_enabled,
            bias_trainable=bias_trainable,
            grid_eps=grid_eps,
            grid_range=grid_range,
            sp_trainable=sp_trainable,
            sb_trainable=sb_trainable,
            device=device,
            seed=seed,
        )

        self.epochs = epochs
        self.learnable_params = learnable_params or {}
        self.optimizer = Adam(self.parameters(), lr=lr)
        logging.debug(f"KANWrapper initialized with parameters: {self.__dict__}")

    def _initialize_parameters(self) -> None:
        """
        Initialize all model parameters and learnable parameters with log-normal distribution.
        """
        try:
            for param in self.parameters():
                param.data = torch.randn_like(param).log_normal_()
            for key, value in self.learnable_params.items():
                if isinstance(value, torch.Tensor):
                    self.learnable_params[key] = torch.randn_like(value).log_normal_()
                else:
                    self.learnable_params[key] = torch.tensor(value).log_normal_()
            logging.debug("All model parameters initialized.")
        except Exception as e:
            logging.error(f"Error initializing parameters: {e}")
            raise

    def _initialize_missing_keys(self) -> None:
        """
        Ensure all required keys are present in learnable_params and initialize them if missing.
        """
        try:
            required_keys = [name for name, _ in self.named_parameters()]
            for key in required_keys:
                if key not in self.learnable_params:
                    self.learnable_params[key] = torch.randn(1).log_normal_()
                    logging.debug(f"Initialized missing key {key}.")
            logging.debug("All missing keys initialized.")
        except Exception as e:
            logging.error(f"Error initializing missing keys: {e}")
            raise

    def save(self, filename: str) -> None:
        """
        Save the network's weights, biases, and learnable parameters to a file.

        Args:
            filename (str): The file path to save the network.
        """
        try:
            state = {
                "model_state_dict": self.state_dict(),
                "learnable_params": self.learnable_params,
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            with open(filename, "wb") as f:
                torch.save(state, f)
            logging.info(f"Model saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load(cls, filename: str) -> "KANWrapper":
        """
        Load a network's weights, biases, and learnable parameters from a file.

        Args:
            filename (str): The file path to load the network from.

        Returns:
            KANWrapper: An instance of the KANWrapper class with loaded weights, biases, and learnable parameters.
        """
        try:
            with open(filename, "rb") as file:
                state = torch.load(file)
            instance = cls(
                width=state["learnable_params"]["width"],
                grid=state["learnable_params"]["grid"],
                k=state["learnable_params"]["k"],
            )
            instance.load_state_dict(state["model_state_dict"])
            instance.optimizer.load_state_dict(state["optimizer_state_dict"])
            instance.learnable_params = state["learnable_params"]
            logging.info(f"Model loaded from {filename}")
            return instance
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def fix_symbolic_function(
        self, l: int, i: int, j: int, expression: str, fit_parameters: bool = True
    ) -> None:
        """Fix a symbolic function in the KAN model."""
        try:
            self.fix_symbolic(l, i, j, expression, fit_parameters)
            logging.info(f"Fixed symbolic function {expression} at ({l}, {i}, {j})")
        except AttributeError as e:
            logging.error(f"Error in fix_symbolic_function: {e}")
            raise

    def get_activation_range(
        self, l: int, i: int, j: int
    ) -> Tuple[float, float, float, float]:
        """
        Get the activation range for a specific neuron.

        Args:
            l (int): Layer index.
                     # Abbreviation: l
                     # Description: The index of the layer in the neural network.
                     # Typical Value: An integer representing the layer number, e.g., 0 for the first layer.
                     # Range: Any valid layer index within the network.

            i (int): Row index.
                     # Abbreviation: i
                     # Description: The index of the row in the layer.
                     # Typical Value: An integer representing the row number, e.g., 0 for the first row.
                     # Range: Any valid row index within the layer.

            j (int): Column index.
                     # Abbreviation: j
                     # Description: The index of the column in the layer.
                     # Typical Value: An integer representing the column number, e.g., 0 for the first column.
                     # Range: Any valid column index within the layer.

        Returns:
            Tuple[float, float, float, float]: The activation range (x_min, x_max, y_min, y_max).
        """
        try:
            x_min, x_max, y_min, y_max = self.get_range(l, i, j)
            logging.info(
                f"Activation range for ({l}, {i}, {j}): x=({x_min}, {x_max}), y=({y_min}, {y_max})"
            )
            return x_min, x_max, y_min, y_max
        except Exception as e:
            logging.error(f"Error getting activation range: {e}")
            raise

    def initialize_from_another_model(
        self, other: "KANWrapper", input_tensor: torch.Tensor
    ) -> None:
        """
        Initialize the model from another parent model.

        Args:
            other (KANWrapper): The parent model to initialize from.
                                # Abbreviation: other
                                # Description: The KANWrapper instance from which to initialize the current model.
                                # Typical Value: An instance of KANWrapper with pre-trained weights.
                                # Range: Any valid KANWrapper instance.

            input_tensor (torch.Tensor): Input tensor for initialization.
                                         # Abbreviation: input_tensor
                                         # Description: The input data used for initializing the model.
                                         # Typical Value: A tensor of shape (batch_size, input_features).
                                         # Range: Any valid tensor shape compatible with the network's input layer.
        """
        try:
            if not isinstance(other, KANWrapper):
                raise TypeError("other must be an instance of KANWrapper")
            if not isinstance(input_tensor, torch.Tensor):
                raise TypeError("input_tensor must be a torch.Tensor")

            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)

            self.initialize_from_another_model(other, input_tensor)
            logging.info("Initialized from another model")

            self._initialize_missing_keys()

            if messagebox.askyesno(
                "Save Initialized Model",
                "Model initialized from another model. Do you want to save the initialized model?",
            ):
                filename = filedialog.asksaveasfilename(
                    defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")]
                )
                if filename:
                    self.save(filename)
                    logging.info(f"Initialized model saved to {filename}")
        except Exception as e:
            logging.error(f"Error initializing from another model: {e}")
            raise

    def mutate(self, mutation_rate: float = 0.1) -> None:
        """
        Mutate the network's weights and biases.

        Args:
            mutation_rate (float): Probability of mutation for each weight and bias.
                                   - Abbreviation: mutation_rate
                                   - Description: The probability that each weight and bias will be mutated.
                                   - Typical Value: 0.1
                                   - Range: [0.0, 1.0]
        """
        try:
            for param in self.parameters():
                if torch.rand(1).item() < mutation_rate:
                    param.data += torch.randn_like(param) * mutation_rate
                    logging.debug(f"Mutated parameter with shape {param.shape}")
            logging.info(f"Applied mutations with mutation rate: {mutation_rate}")
        except Exception as e:
            logging.error(f"Error mutating model parameters: {e}")
            raise

    def inherit(self, other: "KANWrapper") -> None:
        """
        Inherit weights and biases from another KAN instance.

        Args:
            other (KANWrapper): Another KAN instance to inherit from.
                                - Abbreviation: other
                                - Description: The KANWrapper instance from which to inherit weights and biases.
                                - Typical Value: An instance of KANWrapper with pre-trained weights.
                                - Range: Any valid KANWrapper instance.
        """
        try:
            self.load_state_dict(other.state_dict())
            logging.info("Inherited model parameters from another instance.")
        except Exception as e:
            logging.error(f"Error inheriting model parameters: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.
                              - Abbreviation: x
                              - Description: The input data to the network.
                              - Typical Value: A tensor of shape (batch_size, input_features).
                              - Range: Any valid tensor shape compatible with the network's input layer.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        try:
            output = super().forward(x)
            logging.debug(f"Forward pass completed with input shape: {x.shape}")
            return output
        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            raise

    def train_model(
        self,
        train_data: torch.Tensor,  # Training data tensor containing input features.
        train_labels: torch.Tensor,  # Training labels tensor containing target values.
        batch_size: int = 32,  # Batch size for training. Typical value: 32. Range: [1, inf).
        validation_split: float = 0.2,  # Fraction of data to use for validation. Typical value: 0.2. Range: [0, 1].
        shuffle: bool = True,  # Whether to shuffle the data before splitting. Typical value: True.
    ) -> None:
        """
        Train the KAN model.

        Args:
            train_data (torch.Tensor): Training data.
            train_labels (torch.Tensor): Training labels.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of data to use for validation.
            shuffle (bool): Whether to shuffle the data before splitting.
        """
        try:
            train_dataset = TensorDataset(train_data, train_labels)
            train_size = int((1 - validation_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            for epoch in range(self.epochs):
                self.train(
                    dataset=train_loader,  # DataLoader object containing the training data.
                    opt="LBFGS",  # Optimization algorithm to use, "LBFGS" is a quasi-Newton method.
                    steps=100,  # Number of steps to run the optimizer, typical value: 100.
                    log=1,  # Logging frequency, 1 means log every step.
                    lamb=0,  # Regularization parameter lambda, typical value: 0, range: [0, inf).
                    lamb_l1=1,  # L1 regularization parameter, typical value: 1, range: [0, inf).
                    lamb_entropy=2,  # Entropy regularization parameter, typical value: 2, range: [0, inf).
                    lamb_coef=0,  # Coefficient regularization parameter, typical value: 0, range: [0, inf).
                    lamb_coefdiff=0,  # Coefficient difference regularization parameter, typical value: 0, range: [0, inf).
                    update_grid=True,  # Whether to update the grid, typical value: True.
                    grid_update_num=10,  # Number of grid updates, typical value: 10.
                    loss_fn=None,  # Loss function to use, None means default loss function.
                    lr=1,  # Learning rate, typical value: 1, range: (0, inf).
                    stop_grid_update_step=50,  # Step at which to stop grid updates, typical value: 50.
                    batch=-1,  # Batch size, -1 means use the entire dataset as one batch.
                    small_mag_threshold=1e-16,  # Threshold for small magnitude, typical value: 1e-16.
                    small_reg_factor=1,  # Regularization factor for small magnitudes, typical value: 1.
                    metrics=None,  # Metrics to evaluate, None means no additional metrics.
                    sglr_avoid=False,  # Whether to avoid stochastic gradient learning rate, typical value: False.
                    save_fig=False,  # Whether to save figures, typical value: False.
                    in_vars=None,  # Input variables, None means use default input variables.
                    out_vars=None,  # Output variables, None means use default output variables.
                    beta=3,  # Beta parameter for optimization, typical value: 3.
                    save_fig_freq=1,  # Frequency of saving figures, typical value: 1.
                    img_folder="./video",  # Folder to save images, typical value: "./video".
                    device="cpu",  # Device to run the model on, typical value: "cpu".
                )
                epoch_loss = 0.0
                for batch_data, batch_labels in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.forward(batch_data)
                    loss = torch.nn.functional.mse_loss(outputs, batch_labels)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()

                logging.info(
                    f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(train_loader)}"
                )

                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_data, val_labels in val_loader:
                        val_outputs = self.forward(val_data)
                        val_loss += torch.nn.functional.mse_loss(
                            val_outputs, val_labels
                        ).item()
                logging.info(f"Validation Loss: {val_loss/len(val_loader)}")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise


if __name__ == "__main__":
    width = [10, 20, 10]
    grid = 5
    k = 3
    learnable_params = {"lr": 0.01}
    kan_wrapper = KANWrapper(
        width, grid, k, learnable_params=learnable_params, lr=0.01, epochs=10
    )
    train_data = torch.randn(100, 10)
    train_labels = torch.randn(100, 1)
    kan_wrapper.train_model(train_data, train_labels)
