import re
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
def _densenet(growth_rate: int, block_config: Tuple[int, int, int, int], num_init_features: int, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> DenseNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if weights is not None:
        _load_state_dict(model=model, weights=weights, progress=progress)
    return model