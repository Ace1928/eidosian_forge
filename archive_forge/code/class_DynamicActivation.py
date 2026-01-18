import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Callable, Dict, Optional, Union, Any
import logging
class DynamicActivation(nn.Module):
    """
    A custom PyTorch module that dynamically selects and applies an activation function, with detailed logging and error handling.
    """

    def __init__(self, activation_func: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Initialize the DynamicActivation module with a specific activation function.

        Parameters:
            activation_func (Callable[[torch.Tensor], torch.Tensor]): The activation function to be used in the forward pass.
        """
        super(DynamicActivation, self).__init__()
        self.activation_func: Callable[[torch.Tensor], torch.Tensor] = activation_func
        logger.debug(f'DynamicActivation initialized with activation function: {self.activation_func}')

    def forward(self, x: torch.Tensor, batch_id: Optional[int]=None, deterministic: bool=False) -> torch.Tensor:
        """
        Perform the forward pass using the specified activation function, with detailed logging and error handling.

        Parameters:
            x (torch.Tensor): The input tensor to be processed.
            batch_id (Optional[int]): An optional batch identifier for logging purposes.
            deterministic (bool): A flag to determine if the operation should be deterministic.

        Returns:
            torch.Tensor: The activated tensor after applying the activation function.

        Raises:
            RuntimeError: If any exception occurs during the forward pass.
        """
        logger.debug(f'Forward pass initiated for batch_id={batch_id}, deterministic={deterministic}. Input tensor: {x}')
        try:
            output: torch.Tensor = self.activation_func(x)
            logger.debug(f'Output tensor after activation: {output}')
        except Exception as e:
            logger.error(f'An error occurred during the forward pass for batch_id={batch_id}: {e}', exc_info=True)
            raise RuntimeError(f'Forward pass failed due to: {e}') from e
        return output