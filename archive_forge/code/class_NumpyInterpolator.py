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
class NumpyInterpolator(BaseInterpolator):
    """One-dimensional linear interpolation.

    See Also
    --------
    numpy.interp
    """

    def __init__(self, xi, yi, method='linear', fill_value=None, period=None):
        if method != 'linear':
            raise ValueError('only method `linear` is valid for the NumpyInterpolator')
        self.method = method
        self.f = np.interp
        self.cons_kwargs = {}
        self.call_kwargs = {'period': period}
        self._xi = xi
        self._yi = yi
        nan = np.nan if yi.dtype.kind != 'c' else np.nan + np.nan * 1j
        if fill_value is None:
            self._left = nan
            self._right = nan
        elif isinstance(fill_value, Sequence) and len(fill_value) == 2:
            self._left = fill_value[0]
            self._right = fill_value[1]
        elif is_scalar(fill_value):
            self._left = fill_value
            self._right = fill_value
        else:
            raise ValueError(f'{fill_value} is not a valid fill_value')

    def __call__(self, x):
        return self.f(x, self._xi, self._yi, left=self._left, right=self._right, **self.call_kwargs)