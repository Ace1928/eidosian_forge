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
def _mnasnet(alpha: float, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> MNASNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
    model = MNASNet(alpha, **kwargs)
    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model