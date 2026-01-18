import torch
from functorch._C import dim as _C
from . import op_properties
from .batch_tensor import _enable_layers
from .tree_map import tree_flatten, tree_map
import operator
from functools import reduce
def _dims(d, N, keepdim, single_dim):
    from . import Dim
    if isinstance(d, (Dim, int)):
        return ltuple((_wrap_dim(d, N, keepdim),))
    assert not single_dim, f'expected a single dimension or int but found: {d}'
    return ltuple((_wrap_dim(x, N, keepdim) for x in d))