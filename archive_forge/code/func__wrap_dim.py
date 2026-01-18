import torch
from functorch._C import dim as _C
from . import op_properties
from .batch_tensor import _enable_layers
from .tree_map import tree_flatten, tree_map
import operator
from functools import reduce
def _wrap_dim(d, N, keepdim):
    from . import Dim
    if isinstance(d, Dim):
        assert not keepdim, 'cannot preserve first-class dimensions with keepdim=True'
        return d
    elif d >= 0:
        return d - N
    else:
        return d