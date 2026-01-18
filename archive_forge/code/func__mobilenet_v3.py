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
def _mobilenet_v3(inverted_residual_setting: List[InvertedResidualConfig], last_channel: int, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> MobileNetV3:
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model