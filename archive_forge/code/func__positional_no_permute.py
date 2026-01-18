import torch
from functorch._C import dim as _C
from . import op_properties
from .batch_tensor import _enable_layers
from .tree_map import tree_flatten, tree_map
import operator
from functools import reduce
def _positional_no_permute(self, dim, expand_dim=False):
    from . import Tensor
    ptensor, levels = (self._tensor, llist(self._levels))
    try:
        idx = levels.index(dim)
    except ValueError:
        if not expand_dim:
            raise
        idx = 0
        ptensor = ptensor.expand(dim.size, *ptensor.size())
        levels.insert(0, 0)
    idx_batched = 0
    for i in range(idx):
        if isinstance(levels[i], int):
            levels[i] -= 1
            idx_batched += 1
    levels[idx] = -idx_batched - 1
    return (Tensor.from_positional(ptensor, levels, self._has_device), idx_batched)