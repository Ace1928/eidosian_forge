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
def _add_shortcut(x: torch.Tensor, shortcut: torch.Tensor, residual_with_cls_embed: bool):
    if residual_with_cls_embed:
        x.add_(shortcut)
    else:
        x[:, :, 1:, :] += shortcut[:, :, 1:, :]
    return x