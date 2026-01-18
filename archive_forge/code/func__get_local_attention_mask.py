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
def _get_local_attention_mask(attention_mask: torch.Tensor, block_len: int, device: torch.device) -> torch.Tensor:
    """Prepare attention mask to be applied for a local attention."""
    _blocked_attention_mask = _split_into_blocks(attention_mask, block_len, dim=1)
    _3blocked_attention_mask = _concatenate_3_blocks(_blocked_attention_mask, block_dim=1, sequence_dim=2)
    _blocked_attention_mask = _blocked_attention_mask.unsqueeze(-1)
    _3blocked_attention_mask = _3blocked_attention_mask.unsqueeze(-2)
    local_attention_mask = torch.logical_and(_blocked_attention_mask, _3blocked_attention_mask)
    local_attention_mask = _mask_local_attention_mask(local_attention_mask, block_len)
    return local_attention_mask.unsqueeze(1).to(device)