import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import numpy as np
import asyncio
from typing import Dict, Any, List, cast, Tuple, Callable, Optional, Union
import sys
import torch.optim as optim
import os
from ActivationDictionary import ActivationDictionary
from IndegoLogging import configure_logging
class EnhancedPolicyNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, layers_info: List[int], hyperparameters: Optional[Dict[str, float]]=None) -> None:
        """
        Initializes the EnhancedPolicyNetwork with specified dimensions and layer information, including optional hyperparameters.

        This constructor method meticulously sets up a neural network architecture based on the provided input dimension, output dimension,
        and information about the layers. It also allows for the specification of hyperparameters which can include learning rate, regularization factor,
        and discount factor among others. The method ensures that all parameters are validated to prevent any erroneous network configuration.

        The neural network architecture is constructed using the PyTorch framework, leveraging its powerful and flexible modules for building deep learning models.
        The architecture consists of a series of fully connected linear layers, each followed by a rectified linear unit (ReLU) activation function.
        The dimensions of each layer are determined by the `layers_info` parameter, which specifies the number of neurons in each hidden layer.

        The linear layers are created using the `nn.Linear` module from PyTorch, which performs an affine transformation on the input tensor.
        The ReLU activation function is applied after each linear layer using the `nn.ReLU` module, introducing non-linearity into the network.

        Batch normalization is also applied after each ReLU activation using the `nn.BatchNorm1d` module. Batch normalization helps to stabilize the training process
        by normalizing the activations of each layer, reducing the internal covariate shift and allowing for faster convergence.

        The hyperparameters dictionary allows for customization of the learning process, with default values provided for the learning rate, regularization factor,
        and discount factor. These hyperparameters can be adjusted to fine-tune the behavior of the network during training.

        Detailed logging is used throughout the initialization process to trace the network configuration and any potential errors or exceptions that may occur.
        The logging statements provide valuable information for debugging and monitoring the initialization of the EnhancedPolicyNetwork.

        Args:
            input_dim (int): The dimensionality of the input data.
            output_dim (int): The dimensionality of the output data.
            layers_info (List[int]): A list detailing the number of neurons in each hidden layer.
            hyperparameters (Optional[Dict[str, float]]): A dictionary containing learning parameters such as learning rate, regularization factor, etc.

        Raises:
            ValueError: If any of the dimensions or layer information is invalid, specifically if they are non-positive or not provided.

        Returns:
            None
        """
        super(EnhancedPolicyNetwork, self).__init__()
        if input_dim <= 0:
            logger.error(f'Invalid input dimension provided: {input_dim}. Input dimension must be a positive integer.')
            raise ValueError(f'Invalid input dimension provided: {input_dim}. Input dimension must be a positive integer.')
        if output_dim <= 0:
            logger.error(f'Invalid output dimension provided: {output_dim}. Output dimension must be a positive integer.')
            raise ValueError(f'Invalid output dimension provided: {output_dim}. Output dimension must be a positive integer.')
        if not layers_info:
            logger.error('No layer information provided. At least one hidden layer must be specified.')
            raise ValueError('No layer information provided. At least one hidden layer must be specified.')
        if any((layer_dim <= 0 for layer_dim in layers_info)):
            logger.error(f'Invalid layer dimensions provided: {layers_info}. All layer dimensions must be positive integers.')
            raise ValueError(f'Invalid layer dimensions provided: {layers_info}. All layer dimensions must be positive integers.')
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.layers_info: List[int] = layers_info
        if hyperparameters is None:
            self.hyperparameters: Dict[str, float] = {'learning_rate': 0.001, 'regularization_factor': 0.01, 'discount_factor': 0.99}
        else:
            self.hyperparameters: Dict[str, float] = hyperparameters
        logger.debug(f'Initializing EnhancedPolicyNetwork with the following configuration:\n  Input Dimension: {self.input_dim}\n  Output Dimension: {self.output_dim}\n  Layer Information: {self.layers_info}\n  Hyperparameters: {self.hyperparameters}')
        layers: List[nn.Module] = []
        prev_dim: int = self.input_dim
        for layer_dim in self.layers_info:
            linear_layer: nn.Linear = nn.Linear(prev_dim, layer_dim)
            layers.append(linear_layer)
            logger.debug(f'Added linear layer: {linear_layer}')
            relu_layer: nn.ReLU = nn.ReLU()
            layers.append(relu_layer)
            logger.debug(f'Added ReLU activation: {relu_layer}')
            batch_norm_layer: nn.BatchNorm1d = nn.BatchNorm1d(layer_dim)
            layers.append(batch_norm_layer)
            logger.debug(f'Added batch normalization: {batch_norm_layer}')
            prev_dim = layer_dim
        output_layer: nn.Linear = nn.Linear(prev_dim, self.output_dim)
        layers.append(output_layer)
        logger.debug(f'Added output layer: {output_layer}')
        self.model: nn.Sequential = nn.Sequential(*layers)
        logger.debug(f'Created sequential model: {self.model}')
        logger.info('EnhancedPolicyNetwork initialized successfully.')

    def forward(self, x: torch.Tensor, batch_id: Optional[int]=None, deterministic: bool=False) -> torch.Tensor:
        """
        Executes a forward pass through the EnhancedPolicyNetwork with optional determinism.

        This method processes an input tensor `x` through the network's sequential model. If the `deterministic`
        flag is set, it seeds the random number generator to ensure reproducibility. Detailed debug logging is
        provided throughout the execution to trace the computation values and any exceptions encountered.

        Parameters:
            x (torch.Tensor): The input tensor to be processed by the network.
            batch_id (Optional[int]): An optional identifier for the batch being processed, used for logging.
            deterministic (bool): If True, the random number generator is seeded to make the operation deterministic.

        Returns:
            torch.Tensor: The tensor resulting from the forward pass through the network.

        Raises:
            RuntimeError: If any exception occurs during the forward pass, it logs the error and re-raises a
                           RuntimeError with the error message encapsulated.
        """
        logger.debug(f'Forward pass of the EnhancedPolicyNetwork initiated with batch_id={batch_id} and deterministic={deterministic}')
        try:
            x = self.model(x)
            for layer_index, layer in enumerate(self.model[:-1]):
                x = layer(x)
            if deterministic:
                torch.manual_seed(0)
                logger.debug('Random number generator seeded for deterministic operation.')
            logger.debug(f'Final output of the model (Q-values for each action) after processing: {x}')
        except Exception as e:
            logger.error(f'An error occurred during the forward pass: {e}', exc_info=True)
            raise RuntimeError(f'Forward pass failed due to: {e}') from e
        return x