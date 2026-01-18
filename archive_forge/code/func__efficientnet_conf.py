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
def _efficientnet_conf(arch: str, **kwargs: Any) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith('efficientnet_b'):
        bneck_conf = partial(MBConvConfig, width_mult=kwargs.pop('width_mult'), depth_mult=kwargs.pop('depth_mult'))
        inverted_residual_setting = [bneck_conf(1, 3, 1, 32, 16, 1), bneck_conf(6, 3, 2, 16, 24, 2), bneck_conf(6, 5, 2, 24, 40, 2), bneck_conf(6, 3, 2, 40, 80, 3), bneck_conf(6, 5, 1, 80, 112, 3), bneck_conf(6, 5, 2, 112, 192, 4), bneck_conf(6, 3, 1, 192, 320, 1)]
        last_channel = None
    elif arch.startswith('efficientnet_v2_s'):
        inverted_residual_setting = [FusedMBConvConfig(1, 3, 1, 24, 24, 2), FusedMBConvConfig(4, 3, 2, 24, 48, 4), FusedMBConvConfig(4, 3, 2, 48, 64, 4), MBConvConfig(4, 3, 2, 64, 128, 6), MBConvConfig(6, 3, 1, 128, 160, 9), MBConvConfig(6, 3, 2, 160, 256, 15)]
        last_channel = 1280
    elif arch.startswith('efficientnet_v2_m'):
        inverted_residual_setting = [FusedMBConvConfig(1, 3, 1, 24, 24, 3), FusedMBConvConfig(4, 3, 2, 24, 48, 5), FusedMBConvConfig(4, 3, 2, 48, 80, 5), MBConvConfig(4, 3, 2, 80, 160, 7), MBConvConfig(6, 3, 1, 160, 176, 14), MBConvConfig(6, 3, 2, 176, 304, 18), MBConvConfig(6, 3, 1, 304, 512, 5)]
        last_channel = 1280
    elif arch.startswith('efficientnet_v2_l'):
        inverted_residual_setting = [FusedMBConvConfig(1, 3, 1, 32, 32, 4), FusedMBConvConfig(4, 3, 2, 32, 64, 7), FusedMBConvConfig(4, 3, 2, 64, 96, 7), MBConvConfig(4, 3, 2, 96, 192, 10), MBConvConfig(6, 3, 1, 192, 224, 19), MBConvConfig(6, 3, 2, 224, 384, 25), MBConvConfig(6, 3, 1, 384, 640, 7)]
        last_channel = 1280
    else:
        raise ValueError(f'Unsupported model type {arch}')
    return (inverted_residual_setting, last_channel)