import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
def _maxvit(stem_channels: int, block_channels: List[int], block_layers: List[int], stochastic_depth_prob: float, partition_size: int, head_dim: int, weights: Optional[WeightsEnum]=None, progress: bool=False, **kwargs: Any) -> MaxVit:
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
        assert weights.meta['min_size'][0] == weights.meta['min_size'][1]
        _ovewrite_named_param(kwargs, 'input_size', weights.meta['min_size'])
    input_size = kwargs.pop('input_size', (224, 224))
    model = MaxVit(stem_channels=stem_channels, block_channels=block_channels, block_layers=block_layers, stochastic_depth_prob=stochastic_depth_prob, head_dim=head_dim, partition_size=partition_size, input_size=input_size, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model