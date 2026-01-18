from collections.abc import Sequence
import numpy as np
import xarray as xr
from numpy.linalg import LinAlgError
from scipy import special, stats
from . import _remove_indexes_to_reduce
from .linalg import cholesky, eigh
def _broadcast_args(self, args, kwargs):
    """Broadcast and combine initialization and method provided args and kwargs."""
    len_args = len(args) + len(self.args)
    all_args = [*args, *self.args, *kwargs.values(), *self.kwargs.values()]
    broadcastable = []
    non_broadcastable = []
    b_idx = []
    n_idx = []
    for i, a in enumerate(all_args):
        if isinstance(a, xr.DataArray):
            broadcastable.append(a)
            b_idx.append(i)
        else:
            non_broadcastable.append(a)
            n_idx.append(i)
    broadcasted = list(xr.broadcast(*broadcastable))
    all_args = [x for x, _ in sorted(zip(broadcasted + non_broadcastable, b_idx + n_idx), key=lambda pair: pair[1])]
    all_keys = list(kwargs.keys()) + list(self.kwargs.keys())
    args = all_args[:len_args]
    kwargs = dict(zip(all_keys, all_args[len_args:]))
    return (args, kwargs)