import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
class FastSpeech2ConformerConvolutionModule(nn.Module):

    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        super().__init__()
        channels = config.hidden_size
        kernel_size = module_config['kernel_size']
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias=True)
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, hidden_states):
        """
        Compute convolution module.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, time, channels)`): Input tensor.

        Returns:
            `torch.Tensor`: Output tensor of shape `(batch, time, channels)`.

        """
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * torch.sigmoid(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        return hidden_states.transpose(1, 2)