import math
from functools import partial
from typing import Any, Callable, List, Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..ops.misc import MLP, Permute
from ..ops.stochastic_depth import StochasticDepth
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
def _swin_transformer(patch_size: List[int], embed_dim: int, depths: List[int], num_heads: List[int], window_size: List[int], stochastic_depth_prob: float, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> SwinTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
    model = SwinTransformer(patch_size=patch_size, embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, stochastic_depth_prob=stochastic_depth_prob, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model