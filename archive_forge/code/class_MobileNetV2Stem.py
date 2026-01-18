from typing import Optional, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mobilenet_v2 import MobileNetV2Config
class MobileNetV2Stem(nn.Module):

    def __init__(self, config: MobileNetV2Config, in_channels: int, expanded_channels: int, out_channels: int) -> None:
        super().__init__()
        self.first_conv = MobileNetV2ConvLayer(config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=3, stride=2)
        if config.first_layer_is_expansion:
            self.expand_1x1 = None
        else:
            self.expand_1x1 = MobileNetV2ConvLayer(config, in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=1)
        self.conv_3x3 = MobileNetV2ConvLayer(config, in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=3, stride=1, groups=expanded_channels)
        self.reduce_1x1 = MobileNetV2ConvLayer(config, in_channels=expanded_channels, out_channels=out_channels, kernel_size=1, use_activation=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.first_conv(features)
        if self.expand_1x1 is not None:
            features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)
        return features