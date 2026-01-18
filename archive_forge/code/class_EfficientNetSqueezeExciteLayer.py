import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_efficientnet import EfficientNetConfig
class EfficientNetSqueezeExciteLayer(nn.Module):
    """
    This corresponds to the Squeeze and Excitement phase of each block in the original implementation.
    """

    def __init__(self, config: EfficientNetConfig, in_dim: int, expand_dim: int, expand: bool=False):
        super().__init__()
        self.dim = expand_dim if expand else in_dim
        self.dim_se = max(1, int(in_dim * config.squeeze_expansion_ratio))
        self.squeeze = nn.AdaptiveAvgPool2d(output_size=1)
        self.reduce = nn.Conv2d(in_channels=self.dim, out_channels=self.dim_se, kernel_size=1, padding='same')
        self.expand = nn.Conv2d(in_channels=self.dim_se, out_channels=self.dim, kernel_size=1, padding='same')
        self.act_reduce = ACT2FN[config.hidden_act]
        self.act_expand = nn.Sigmoid()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        inputs = hidden_states
        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.reduce(hidden_states)
        hidden_states = self.act_reduce(hidden_states)
        hidden_states = self.expand(hidden_states)
        hidden_states = self.act_expand(hidden_states)
        hidden_states = torch.mul(inputs, hidden_states)
        return hidden_states