from functools import partial
from typing import Any, Callable, Optional
import torch
from torch import nn
from torchvision.ops.misc import Conv3dNormActivation
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
class SepInceptionBlock3D(nn.Module):

    def __init__(self, in_planes: int, b0_out: int, b1_mid: int, b1_out: int, b2_mid: int, b2_out: int, b3_out: int, norm_layer: Callable[..., nn.Module]):
        super().__init__()
        self.branch0 = Conv3dNormActivation(in_planes, b0_out, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.branch1 = nn.Sequential(Conv3dNormActivation(in_planes, b1_mid, kernel_size=1, stride=1, norm_layer=norm_layer), TemporalSeparableConv(b1_mid, b1_out, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer))
        self.branch2 = nn.Sequential(Conv3dNormActivation(in_planes, b2_mid, kernel_size=1, stride=1, norm_layer=norm_layer), TemporalSeparableConv(b2_mid, b2_out, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer))
        self.branch3 = nn.Sequential(nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1), Conv3dNormActivation(in_planes, b3_out, kernel_size=1, stride=1, norm_layer=norm_layer))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out