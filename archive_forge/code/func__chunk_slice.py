import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from .tensor_utils import tensor_tree_map, tree_map
@torch.jit.ignore
def _chunk_slice(t: torch.Tensor, flat_start: int, flat_end: int, no_batch_dims: int) -> torch.Tensor:
    """
    Equivalent to

        t.reshape((-1,) + t.shape[no_batch_dims:])[flat_start:flat_end]

    but without the need for the initial reshape call, which can be memory-intensive in certain situations. The only
    reshape operations in this function are performed on sub-tensors that scale with (flat_end - flat_start), the chunk
    size.
    """
    batch_dims = t.shape[:no_batch_dims]
    start_idx = list(_flat_idx_to_idx(flat_start, batch_dims))
    end_idx = list(_flat_idx_to_idx(flat_end - 1, batch_dims))
    slices = _get_minimal_slice_set(start_idx, end_idx, batch_dims)
    sliced_tensors = [t[s] for s in slices]
    return torch.cat([s.view((-1,) + t.shape[no_batch_dims:]) for s in sliced_tensors])