from __future__ import annotations
import datetime as dt
import warnings
from collections.abc import Hashable, Sequence
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, get_args
import numpy as np
import pandas as pd
from xarray.core import utils
from xarray.core.common import _contains_datetime_like_objects, ones_like
from xarray.core.computation import apply_ufunc
from xarray.core.duck_array_ops import (
from xarray.core.options import _get_keep_attrs
from xarray.core.types import Interp1dOptions, InterpOptions
from xarray.core.utils import OrderedSet, is_scalar
from xarray.core.variable import Variable, broadcast_variables
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def _chunked_aware_interpnd(var, *coords, interp_func, interp_kwargs, localize=True):
    """Wrapper for `_interpnd` through `blockwise` for chunked arrays.

    The first half arrays in `coords` are original coordinates,
    the other half are destination coordinates
    """
    n_x = len(coords) // 2
    nconst = len(var.shape) - n_x
    x = [Variable([f'dim_{nconst + dim}'], _x) for dim, _x in enumerate(coords[:n_x])]
    new_x = [Variable([f'dim_{len(var.shape) + dim}' for dim in range(len(_x.shape))], _x) for _x in coords[n_x:]]
    if localize:
        var = Variable([f'dim_{dim}' for dim in range(len(var.shape))], var)
        indexes_coords = {_x.dims[0]: (_x, _new_x) for _x, _new_x in zip(x, new_x)}
        var, indexes_coords = _localize(var, indexes_coords)
        x, new_x = zip(*[indexes_coords[d] for d in indexes_coords])
        var = var.data
    return _interpnd(var, x, new_x, interp_func, interp_kwargs)