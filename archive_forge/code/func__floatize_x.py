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
def _floatize_x(x, new_x):
    """Make x and new_x float.
    This is particularly useful for datetime dtype.
    x, new_x: tuple of np.ndarray
    """
    x = list(x)
    new_x = list(new_x)
    for i in range(len(x)):
        if _contains_datetime_like_objects(x[i]):
            xmin = x[i].values.min()
            x[i] = x[i]._to_numeric(offset=xmin, dtype=np.float64)
            new_x[i] = new_x[i]._to_numeric(offset=xmin, dtype=np.float64)
    return (x, new_x)