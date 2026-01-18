from functools import partial
from typing import Any, Callable, List, Optional, Sequence
import torch
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation as SElayer
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
def _mobilenet_v3_conf(arch: str, width_mult: float=1.0, reduced_tail: bool=False, dilated: bool=False, **kwargs: Any):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)
    if arch == 'mobilenet_v3_large':
        inverted_residual_setting = [bneck_conf(16, 3, 16, 16, False, 'RE', 1, 1), bneck_conf(16, 3, 64, 24, False, 'RE', 2, 1), bneck_conf(24, 3, 72, 24, False, 'RE', 1, 1), bneck_conf(24, 5, 72, 40, True, 'RE', 2, 1), bneck_conf(40, 5, 120, 40, True, 'RE', 1, 1), bneck_conf(40, 5, 120, 40, True, 'RE', 1, 1), bneck_conf(40, 3, 240, 80, False, 'HS', 2, 1), bneck_conf(80, 3, 200, 80, False, 'HS', 1, 1), bneck_conf(80, 3, 184, 80, False, 'HS', 1, 1), bneck_conf(80, 3, 184, 80, False, 'HS', 1, 1), bneck_conf(80, 3, 480, 112, True, 'HS', 1, 1), bneck_conf(112, 3, 672, 112, True, 'HS', 1, 1), bneck_conf(112, 5, 672, 160 // reduce_divider, True, 'HS', 2, dilation), bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, 'HS', 1, dilation), bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, 'HS', 1, dilation)]
        last_channel = adjust_channels(1280 // reduce_divider)
    elif arch == 'mobilenet_v3_small':
        inverted_residual_setting = [bneck_conf(16, 3, 16, 16, True, 'RE', 2, 1), bneck_conf(16, 3, 72, 24, False, 'RE', 2, 1), bneck_conf(24, 3, 88, 24, False, 'RE', 1, 1), bneck_conf(24, 5, 96, 40, True, 'HS', 2, 1), bneck_conf(40, 5, 240, 40, True, 'HS', 1, 1), bneck_conf(40, 5, 240, 40, True, 'HS', 1, 1), bneck_conf(40, 5, 120, 48, True, 'HS', 1, 1), bneck_conf(48, 5, 144, 48, True, 'HS', 1, 1), bneck_conf(48, 5, 288, 96 // reduce_divider, True, 'HS', 2, dilation), bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, 'HS', 1, dilation), bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, 'HS', 1, dilation)]
        last_channel = adjust_channels(1024 // reduce_divider)
    else:
        raise ValueError(f'Unsupported model type {arch}')
    return (inverted_residual_setting, last_channel)