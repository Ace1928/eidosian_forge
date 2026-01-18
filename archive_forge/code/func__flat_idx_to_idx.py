import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from .tensor_utils import tensor_tree_map, tree_map
@torch.jit.ignore
def _flat_idx_to_idx(flat_idx: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
    idx = []
    for d in reversed(dims):
        idx.append(flat_idx % d)
        flat_idx = flat_idx // d
    return tuple(reversed(idx))