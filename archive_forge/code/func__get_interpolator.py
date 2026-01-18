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
def _get_interpolator(method: InterpOptions, vectorizeable_only: bool=False, **kwargs):
    """helper function to select the appropriate interpolator class

    returns interpolator class and keyword arguments for the class
    """
    interp_class: type[NumpyInterpolator] | type[ScipyInterpolator] | type[SplineInterpolator]
    interp1d_methods = get_args(Interp1dOptions)
    valid_methods = tuple((vv for v in get_args(InterpOptions) for vv in get_args(v)))
    if method == 'linear' and (not kwargs.get('fill_value', None) == 'extrapolate') and (not vectorizeable_only):
        kwargs.update(method=method)
        interp_class = NumpyInterpolator
    elif method in valid_methods:
        if method in interp1d_methods:
            kwargs.update(method=method)
            interp_class = ScipyInterpolator
        elif vectorizeable_only:
            raise ValueError(f'{method} is not a vectorizeable interpolator. Available methods are {interp1d_methods}')
        elif method == 'barycentric':
            interp_class = _import_interpolant('BarycentricInterpolator', method)
        elif method in ['krogh', 'krog']:
            interp_class = _import_interpolant('KroghInterpolator', method)
        elif method == 'pchip':
            interp_class = _import_interpolant('PchipInterpolator', method)
        elif method == 'spline':
            kwargs.update(method=method)
            interp_class = SplineInterpolator
        elif method == 'akima':
            interp_class = _import_interpolant('Akima1DInterpolator', method)
        else:
            raise ValueError(f'{method} is not a valid scipy interpolator')
    else:
        raise ValueError(f'{method} is not a valid interpolator')
    return (interp_class, kwargs)