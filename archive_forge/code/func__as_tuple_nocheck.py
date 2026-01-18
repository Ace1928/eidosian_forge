from typing import List, Tuple
import torch
from torch._vmap_internals import _vmap
from . import forward_ad as fwAD
def _as_tuple_nocheck(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)