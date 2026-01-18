import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, width_in: int, width_out: int, stride: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], group_width: int=1, bottleneck_multiplier: float=1.0, se_ratio: Optional[float]=None) -> None:
        super().__init__()
        self.proj = None
        should_proj = width_in != width_out or stride != 1
        if should_proj:
            self.proj = Conv2dNormActivation(width_in, width_out, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None)
        self.f = BottleneckTransform(width_in, width_out, stride, norm_layer, activation_layer, group_width, bottleneck_multiplier, se_ratio)
        self.activation = activation_layer(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)