from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.ops import Conv2dNormActivation
from ...transforms._presets import OpticalFlow
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._utils import handle_legacy_interface
from ._utils import grid_sample, make_coords_grid, upsample_flow
class ResidualBlock(nn.Module):
    """Slightly modified Residual block with extra relu and biases."""

    def __init__(self, in_channels, out_channels, *, norm_layer, stride=1, always_project: bool=False):
        super().__init__()
        self.convnormrelu1 = Conv2dNormActivation(in_channels, out_channels, norm_layer=norm_layer, kernel_size=3, stride=stride, bias=True)
        self.convnormrelu2 = Conv2dNormActivation(out_channels, out_channels, norm_layer=norm_layer, kernel_size=3, bias=True)
        self.downsample: nn.Module
        if stride == 1 and (not always_project):
            self.downsample = nn.Identity()
        else:
            self.downsample = Conv2dNormActivation(in_channels, out_channels, norm_layer=norm_layer, kernel_size=1, stride=stride, bias=True, activation_layer=None)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = x
        y = self.convnormrelu1(y)
        y = self.convnormrelu2(y)
        x = self.downsample(x)
        return self.relu(x + y)