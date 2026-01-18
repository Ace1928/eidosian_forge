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
class TemporalSeparableConv(nn.Sequential):

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int, padding: int, norm_layer: Callable[..., nn.Module]):
        super().__init__(Conv3dNormActivation(in_planes, out_planes, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=(0, padding, padding), bias=False, norm_layer=norm_layer), Conv3dNormActivation(out_planes, out_planes, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False, norm_layer=norm_layer))