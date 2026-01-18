from typing import Any, List, Mapping, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from .attn_bias import AttentionBias
from .common import (
def _get_tensor_bias_bmk(attn_bias: Optional[Union[torch.Tensor, AttentionBias]]) -> Optional[torch.Tensor]:
    if not isinstance(attn_bias, torch.Tensor):
        assert attn_bias is None
        return None
    if attn_bias.ndim == 4:
        attn_bias = attn_bias.reshape([-1, *attn_bias.shape[2:]])
    return attn_bias