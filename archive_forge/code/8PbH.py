import torch
import numpy as np
import random
import pickle
from typing import Callable, List, Tuple
import kan
from kan import KAN


class KAN:
    def __init__(self, width: List[int], grid: int, k: int):
        """
        Initialize the Kolmogorov-Arnold Network (KAN).

        Args:
            width (List[int]): List of integers specifying the width of each layer.
            grid (int): Grid size for the KAN.
            k (int): Number of neurons in the hidden layer.
        """
        self.model = PyKAN(width=width, grid=grid, k=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        return self.model(x)

    def mutate(self, mutation_rate: float = 0.1):
        """
        Mutate the network's weights and biases.

        Args:
            mutation_rate (float): Probability of mutation for each weight and bias.
        """
        for param in self.model.parameters():
            if random.random() < mutation_rate:
                param.data += torch.randn_like(param) * mutation_rate

    def inherit(self, other: "KAN"):
        """
        Inherit weights and biases from another KAN instance.

        Args:
            other (KAN): Another KAN instance to inherit from.
        """
        self.model.load_state_dict(other.model.state_dict())

    def save(self, filename: str):
        """
        Save the network's weights and biases to a file.

        Args:
            filename (str): The file path to save the network.
        """
        torch.save(self.model.state_dict(), filename)

    @classmethod
    def load(cls, filename: str, width: List[int], grid: int, k: int) -> "KAN":
        """
        Load a network's weights and biases from a file.

        Args:
            filename (str): The file path to load the network from.
            width (List[int]): List of integers specifying the width of each layer.
            grid (int): Grid size for the KAN.
            k (int): Number of neurons in the hidden layer.

        Returns:
            KAN: An instance of the KAN class with loaded weights and biases.
        """
        instance = cls(width=width, grid=grid, k=k)
        instance.model.load_state_dict(torch.load(filename))
        return instance

    def train(
        self,
        dataset: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 0.01,
        epochs: int = 100,
    ):
        """
        Train the network using a simple gradient descent algorithm.

        Args:
            dataset (torch.Tensor): Input data for training.
            targets (torch.Tensor): Target outputs for training.
            learning_rate (float): Learning rate for gradient descent.
            epochs (int): Number of training epochs.
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            for x, y in zip(dataset, targets):
                optimizer.zero_grad()
                output = self.forward(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()

    def initialize_from_another_model(self, other: "KAN", x: torch.Tensor):
        """
        Initialize the model from another KAN instance.

        Args:
            other (KAN): Another KAN instance to initialize from.
            x (torch.Tensor): Input tensor for forward pass.
        """
        self.model.initialize_from_another_model(other.model, x)


if __name__ == "__main__":
    # Example usage of the KAN class
    width = [2, 5, 3]
    grid = 5
    k = 3

    # Initialize the KAN
    kan_instance = KAN(width=width, grid=grid, k=k)

    # Create dummy data for training
    dataset = torch.normal(0, 1, size=(100, 2))
    targets = torch.normal(0, 1, size=(100, 3))

    # Train the KAN
    kan_instance.train(dataset, targets, learning_rate=0.01, epochs=10)

    # Save the trained model
    kan_instance.save("kan_model.pth")

    # Load the model
    loaded_kan = KAN.load("kan_model.pth", width=width, grid=grid, k=k)

    # Perform a forward pass with the loaded model
    test_input = torch.normal(0, 1, size=(1, 2))
    output = loaded_kan.forward(test_input)
    print(f"Output for test input: {output}")

    # Initialize a fine model from a coarse model
    model_fine = KAN(width=[2, 5, 1], grid=10, k=3)
    model_fine.initialize_from_another_model(loaded_kan, test_input)
    print(
        f"Output for test input after fine initialization: {model_fine.forward(test_input)}"
    )
