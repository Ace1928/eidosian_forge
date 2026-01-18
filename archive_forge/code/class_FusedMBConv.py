import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
class FusedMBConv(nn.Module):

    def __init__(self, cnf: FusedMBConvConfig, stochastic_depth_prob: float, norm_layer: Callable[..., nn.Module]) -> None:
        super().__init__()
        if not 1 <= cnf.stride <= 2:
            raise ValueError('illegal stride value')
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers: List[nn.Module] = []
        activation_layer = nn.SiLU
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(Conv2dNormActivation(cnf.input_channels, expanded_channels, kernel_size=cnf.kernel, stride=cnf.stride, norm_layer=norm_layer, activation_layer=activation_layer))
            layers.append(Conv2dNormActivation(expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None))
        else:
            layers.append(Conv2dNormActivation(cnf.input_channels, cnf.out_channels, kernel_size=cnf.kernel, stride=cnf.stride, norm_layer=norm_layer, activation_layer=activation_layer))
        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, 'row')
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result