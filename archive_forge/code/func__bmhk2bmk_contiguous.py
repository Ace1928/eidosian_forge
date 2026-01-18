from typing import Any, List, Mapping, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from .attn_bias import AttentionBias
from .common import (
def _bmhk2bmk_contiguous(tensor) -> torch.Tensor:
    return tensor.permute((0, 2, 1, 3)).contiguous().view([tensor.shape[0] * tensor.shape[2], tensor.shape[1], tensor.shape[3]]).contiguous()