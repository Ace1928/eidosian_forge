import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
class FastSpeech2ConformerMultiLayeredConv1d(nn.Module):
    """
    Multi-layered conv1d for Transformer block.

    This is a module of multi-layered conv1d designed to replace positionwise feed-forward network in Transformer
    block, which is introduced in 'FastSpeech: Fast, Robust and Controllable Text to Speech'
    https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        """
        Initialize FastSpeech2ConformerMultiLayeredConv1d module.

        Args:
            input_channels (`int`): Number of input channels.
            hidden_channels (`int`): Number of hidden channels.
            kernel_size (`int`): Kernel size of conv1d.
            dropout_rate (`float`): Dropout rate.
        """
        super().__init__()
        input_channels = config.hidden_size
        hidden_channels = module_config['linear_units']
        kernel_size = config.positionwise_conv_kernel_size
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(hidden_channels, input_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = nn.Dropout(module_config['dropout_rate'])

    def forward(self, hidden_states):
        """
        Calculate forward propagation.

        Args:
            hidden_states (torch.Tensor): Batch of input tensors (batch_size, time, input_channels).

        Returns:
            torch.Tensor: Batch of output tensors (batch_size, time, hidden_channels).
        """
        hidden_states = hidden_states.transpose(-1, 1)
        hidden_states = self.conv1(hidden_states)
        hidden_states = torch.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.transpose(-1, 1)
        return hidden_states