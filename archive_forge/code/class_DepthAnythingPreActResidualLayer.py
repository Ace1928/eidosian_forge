from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...file_utils import (
from ...modeling_outputs import DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..auto import AutoBackbone
from .configuration_depth_anything import DepthAnythingConfig
class DepthAnythingPreActResidualLayer(nn.Module):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[DepthAnythingConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()
        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)
        return hidden_state + residual