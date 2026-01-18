import warnings
from functools import partial
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class _InvertedResidual(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, expansion_factor: int, bn_momentum: float=0.1) -> None:
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f'stride should be 1 or 2 instead of {stride}')
        if kernel_size not in [3, 5]:
            raise ValueError(f'kernel_size should be 3 or 5 instead of {kernel_size}')
        mid_ch = in_ch * expansion_factor
        self.apply_residual = in_ch == out_ch and stride == 1
        self.layers = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch, momentum=bn_momentum), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2, stride=stride, groups=mid_ch, bias=False), nn.BatchNorm2d(mid_ch, momentum=bn_momentum), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch, momentum=bn_momentum))

    def forward(self, input: Tensor) -> Tensor:
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)