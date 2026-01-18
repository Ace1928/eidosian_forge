from typing import Optional, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mobilenet_v2 import MobileNetV2Config
class MobileNetV2InvertedResidual(nn.Module):

    def __init__(self, config: MobileNetV2Config, in_channels: int, out_channels: int, stride: int, dilation: int=1) -> None:
        super().__init__()
        expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), config.depth_divisible_by, config.min_depth)
        if stride not in [1, 2]:
            raise ValueError(f'Invalid stride {stride}.')
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand_1x1 = MobileNetV2ConvLayer(config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1)
        self.conv_3x3 = MobileNetV2ConvLayer(config, in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=3, stride=stride, groups=expanded_channels, dilation=dilation)
        self.reduce_1x1 = MobileNetV2ConvLayer(config, in_channels=expanded_channels, out_channels=out_channels, kernel_size=1, use_activation=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)
        return residual + features if self.use_residual else features