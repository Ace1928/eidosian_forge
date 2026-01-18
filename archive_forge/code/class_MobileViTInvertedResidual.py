import math
from typing import Dict, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mobilevit import MobileViTConfig
class MobileViTInvertedResidual(nn.Module):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int, dilation: int=1) -> None:
        super().__init__()
        expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)
        if stride not in [1, 2]:
            raise ValueError(f'Invalid stride {stride}.')
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand_1x1 = MobileViTConvLayer(config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1)
        self.conv_3x3 = MobileViTConvLayer(config, in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=3, stride=stride, groups=expanded_channels, dilation=dilation)
        self.reduce_1x1 = MobileViTConvLayer(config, in_channels=expanded_channels, out_channels=out_channels, kernel_size=1, use_activation=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)
        return residual + features if self.use_residual else features