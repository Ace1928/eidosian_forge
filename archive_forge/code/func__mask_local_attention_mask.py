import copy
import math
import warnings
from typing import Any, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_longt5 import LongT5Config
def _mask_local_attention_mask(local_attention_mask: torch.Tensor, block_len: int) -> torch.Tensor:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_radius."""
    relative_position_ids = _make_3block_relative_position_ids(block_len)
    locality_mask = torch.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    locality_mask = locality_mask.to(local_attention_mask.device)
    return torch.logical_and(local_attention_mask, locality_mask)