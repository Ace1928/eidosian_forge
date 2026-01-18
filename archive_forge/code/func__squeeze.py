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
def _squeeze(x: torch.Tensor, target_dim: int, expand_dim: int, tensor_dim: int) -> torch.Tensor:
    if tensor_dim == target_dim - 1:
        x = x.squeeze(expand_dim)
    return x