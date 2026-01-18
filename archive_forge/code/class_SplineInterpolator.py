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
class SplineInterpolator(BaseInterpolator):
    """One-dimensional smoothing spline fit to a given set of data points.

    See Also
    --------
    scipy.interpolate.UnivariateSpline
    """

    def __init__(self, xi, yi, method='spline', fill_value=None, order=3, nu=0, ext=None, **kwargs):
        from scipy.interpolate import UnivariateSpline
        if method != 'spline':
            raise ValueError('only method `spline` is valid for the SplineInterpolator')
        self.method = method
        self.cons_kwargs = kwargs
        self.call_kwargs = {'nu': nu, 'ext': ext}
        if fill_value is not None:
            raise ValueError('SplineInterpolator does not support fill_value')
        self.f = UnivariateSpline(xi, yi, k=order, **self.cons_kwargs)