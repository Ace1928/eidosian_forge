import warnings
from typing import Iterable, List, NamedTuple, Tuple, Union
import torch
from torch import Tensor
from ... import _VF
from ..._jit_internal import Optional
class PackedSequence_(NamedTuple):
    data: torch.Tensor
    batch_sizes: torch.Tensor
    sorted_indices: Optional[torch.Tensor]
    unsorted_indices: Optional[torch.Tensor]