from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..swin_transformer import PatchMerging, SwinTransformerBlock
def _get_relative_position_bias(relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]) -> Tensor:
    window_vol = window_size[0] * window_size[1] * window_size[2]
    relative_position_bias = relative_position_bias_table[relative_position_index[:window_vol, :window_vol].flatten()]
    relative_position_bias = relative_position_bias.view(window_vol, window_vol, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias