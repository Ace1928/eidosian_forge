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
def _concatenate_3_blocks(x: torch.Tensor, block_dim: int, sequence_dim: int, pad_value: int=0) -> torch.Tensor:
    """Concatenate three consecutive blocks for each input block for local attentiont.

    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_dim]
    pad = [(0, 0)] * x.ndim
    pad[block_dim] = (1, 1)
    pad = sum(pad[::-1], ())
    x = nn.functional.pad(x, pad=pad, mode='constant', value=pad_value)
    blocks_list: List[torch.Tensor] = []
    for i in range(3):
        indices = [slice(0, None)] * x.ndim
        indices[block_dim] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    return torch.cat(blocks_list, dim=sequence_dim)