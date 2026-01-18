from __future__ import annotations
import functools
from typing import Optional
import torch
from . import _dtypes_impl, _util
from ._normalizations import (
def _deco_axis_expand(func):
    """
    Generically handle axis arguments in reductions.
    axis is *always* the 2nd arg in the function so no need to have a look at its signature
    """

    @functools.wraps(func)
    def wrapped(a, axis=None, *args, **kwds):
        if axis is not None:
            axis = _util.normalize_axis_tuple(axis, a.ndim)
        if axis == ():
            newshape = _util.expand_shape(a.shape, axis=0)
            a = a.reshape(newshape)
            axis = (0,)
        return func(a, axis, *args, **kwds)
    return wrapped