from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_convnextv2 import ConvNextV2Config
class ConvNextV2Stage(nn.Module):
    """ConvNeXTV2 stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config ([`ConvNextV2Config`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`List[float]`): Stochastic depth rates for each layer.
    """

    def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
        super().__init__()
        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = nn.Sequential(ConvNextV2LayerNorm(in_channels, eps=1e-06, data_format='channels_first'), nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
        else:
            self.downsampling_layer = nn.Identity()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = nn.Sequential(*[ConvNextV2Layer(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)])

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        hidden_states = self.downsampling_layer(hidden_states)
        hidden_states = self.layers(hidden_states)
        return hidden_states