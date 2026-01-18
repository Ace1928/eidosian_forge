from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
class SuperPointConvBlock(nn.Module):

    def __init__(self, config: SuperPointConfig, in_channels: int, out_channels: int, add_pooling: bool=False) -> None:
        super().__init__()
        self.conv_a = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_b = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if add_pooling else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.relu(self.conv_a(hidden_states))
        hidden_states = self.relu(self.conv_b(hidden_states))
        if self.pool is not None:
            hidden_states = self.pool(hidden_states)
        return hidden_states