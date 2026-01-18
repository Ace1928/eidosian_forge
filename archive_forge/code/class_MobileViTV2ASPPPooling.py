from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mobilevitv2 import MobileViTV2Config
class MobileViTV2ASPPPooling(nn.Module):

    def __init__(self, config: MobileViTV2Config, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv_1x1 = MobileViTV2ConvLayer(config, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, use_normalization=True, use_activation='relu')

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial_size = features.shape[-2:]
        features = self.global_pool(features)
        features = self.conv_1x1(features)
        features = nn.functional.interpolate(features, size=spatial_size, mode='bilinear', align_corners=False)
        return features