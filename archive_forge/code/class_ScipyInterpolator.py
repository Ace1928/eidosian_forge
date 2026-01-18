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
class ScipyInterpolator(BaseInterpolator):
    """Interpolate a 1-D function using Scipy interp1d

    See Also
    --------
    scipy.interpolate.interp1d
    """

    def __init__(self, xi, yi, method=None, fill_value=None, assume_sorted=True, copy=False, bounds_error=False, order=None, **kwargs):
        from scipy.interpolate import interp1d
        if method is None:
            raise ValueError('method is a required argument, please supply a valid scipy.inter1d method (kind)')
        if method == 'polynomial':
            if order is None:
                raise ValueError('order is required when method=polynomial')
            method = order
        self.method = method
        self.cons_kwargs = kwargs
        self.call_kwargs = {}
        nan = np.nan if yi.dtype.kind != 'c' else np.nan + np.nan * 1j
        if fill_value is None and method == 'linear':
            fill_value = (nan, nan)
        elif fill_value is None:
            fill_value = nan
        self.f = interp1d(xi, yi, kind=self.method, fill_value=fill_value, bounds_error=bounds_error, assume_sorted=assume_sorted, copy=copy, **self.cons_kwargs)