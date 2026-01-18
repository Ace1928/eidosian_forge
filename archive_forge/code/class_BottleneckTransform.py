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
class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, width_in: int, width_out: int, stride: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], group_width: int, bottleneck_multiplier: float, se_ratio: Optional[float]) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width
        layers['a'] = Conv2dNormActivation(width_in, w_b, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=activation_layer)
        layers['b'] = Conv2dNormActivation(w_b, w_b, kernel_size=3, stride=stride, groups=g, norm_layer=norm_layer, activation_layer=activation_layer)
        if se_ratio:
            width_se_out = int(round(se_ratio * width_in))
            layers['se'] = SqueezeExcitation(input_channels=w_b, squeeze_channels=width_se_out, activation=activation_layer)
        layers['c'] = Conv2dNormActivation(w_b, width_out, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=None)
        super().__init__(layers)