import torch
from functorch._C import dim as _C
from . import op_properties
from .batch_tensor import _enable_layers
from .tree_map import tree_flatten, tree_map
import operator
from functools import reduce
def _tensor_levels(inp):
    from . import _Tensor
    if isinstance(inp, _Tensor):
        return (inp._tensor, llist(inp._levels), inp._has_device)
    else:
        return (inp, llist(range(-inp.ndim, 0)), True)