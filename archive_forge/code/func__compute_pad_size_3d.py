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
def _compute_pad_size_3d(size_dhw: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    pad_size = [(patch_size[i] - size_dhw[i] % patch_size[i]) % patch_size[i] for i in range(3)]
    return (pad_size[0], pad_size[1], pad_size[2])