from typing import Optional
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_regnet import RegNetConfig
class RegNetSELayer(nn.Module):
    """
    Squeeze and Excitation layer (SE) proposed in [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).
    """

    def __init__(self, in_channels: int, reduced_channels: int):
        super().__init__()
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.attention = nn.Sequential(nn.Conv2d(in_channels, reduced_channels, kernel_size=1), nn.ReLU(), nn.Conv2d(reduced_channels, in_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, hidden_state):
        pooled = self.pooler(hidden_state)
        attention = self.attention(pooled)
        hidden_state = hidden_state * attention
        return hidden_state