import math
from collections import OrderedDict
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....activations import ACT2FN
from ....modeling_outputs import (
from ....modeling_utils import PreTrainedModel
from ....utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_van import VanConfig
class VanMlpLayer(nn.Module):
    """
    MLP with depth-wise convolution, from [PVTv2: Improved Baselines with Pyramid Vision
    Transformer](https://arxiv.org/abs/2106.13797).
    """

    def __init__(self, in_channels: int, hidden_size: int, out_channels: int, hidden_act: str='gelu', dropout_rate: float=0.5):
        super().__init__()
        self.in_dense = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        self.depth_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)
        self.activation = ACT2FN[hidden_act]
        self.dropout1 = nn.Dropout(dropout_rate)
        self.out_dense = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.in_dense(hidden_state)
        hidden_state = self.depth_wise(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout1(hidden_state)
        hidden_state = self.out_dense(hidden_state)
        hidden_state = self.dropout2(hidden_state)
        return hidden_state