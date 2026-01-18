import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import torch
import torch.fx
import torch.nn as nn
from ...ops import MLP, StochasticDepth
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
def _mvit(block_setting: List[MSBlockConfig], stochastic_depth_prob: float, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> MViT:
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
        assert weights.meta['min_size'][0] == weights.meta['min_size'][1]
        _ovewrite_named_param(kwargs, 'spatial_size', weights.meta['min_size'])
        _ovewrite_named_param(kwargs, 'temporal_size', weights.meta['min_temporal_size'])
    spatial_size = kwargs.pop('spatial_size', (224, 224))
    temporal_size = kwargs.pop('temporal_size', 16)
    model = MViT(spatial_size=spatial_size, temporal_size=temporal_size, block_setting=block_setting, residual_pool=kwargs.pop('residual_pool', False), residual_with_cls_embed=kwargs.pop('residual_with_cls_embed', True), rel_pos_embed=kwargs.pop('rel_pos_embed', False), proj_after_attn=kwargs.pop('proj_after_attn', False), stochastic_depth_prob=stochastic_depth_prob, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model