from functools import partial
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import shufflenetv2
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..shufflenetv2 import (
from .utils import _fuse_modules, _replace_relu, quantize_model
def _shufflenetv2(stages_repeats: List[int], stages_out_channels: List[int], *, weights: Optional[WeightsEnum], progress: bool, quantize: bool, **kwargs: Any) -> QuantizableShuffleNetV2:
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
        if 'backend' in weights.meta:
            _ovewrite_named_param(kwargs, 'backend', weights.meta['backend'])
    backend = kwargs.pop('backend', 'fbgemm')
    model = QuantizableShuffleNetV2(stages_repeats, stages_out_channels, **kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model