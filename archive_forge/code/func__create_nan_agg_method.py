from __future__ import annotations
import contextlib
import datetime
import inspect
import warnings
from functools import partial
from importlib import import_module
import numpy as np
import pandas as pd
from numpy import all as array_all  # noqa
from numpy import any as array_any  # noqa
from numpy import (  # noqa
from numpy import concatenate as _concatenate
from numpy.lib.stride_tricks import sliding_window_view  # noqa
from packaging.version import Version
from xarray.core import dask_array_ops, dtypes, nputils
from xarray.core.options import OPTIONS
from xarray.core.utils import is_duck_array, is_duck_dask_array, module_available
from xarray.namedarray import pycompat
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, is_chunked_array
def _create_nan_agg_method(name, coerce_strings=False, invariant_0d=False):
    from xarray.core import nanops

    def f(values, axis=None, skipna=None, **kwargs):
        if kwargs.pop('out', None) is not None:
            raise TypeError(f'`out` is not valid for {name}')
        if invariant_0d and axis == ():
            return values
        values = asarray(values)
        if coerce_strings and values.dtype.kind in 'SU':
            values = astype(values, object)
        func = None
        if skipna or (skipna is None and values.dtype.kind in 'cfO'):
            nanname = 'nan' + name
            func = getattr(nanops, nanname)
        else:
            if name in ['sum', 'prod']:
                kwargs.pop('min_count', None)
            xp = get_array_namespace(values)
            func = getattr(xp, name)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'All-NaN slice encountered')
                return func(values, axis=axis, **kwargs)
        except AttributeError:
            if not is_duck_dask_array(values):
                raise
            try:
                return func(values, axis=axis, dtype=values.dtype, **kwargs)
            except (AttributeError, TypeError):
                raise NotImplementedError(f'{name} is not yet implemented on dask arrays')
    f.__name__ = name
    return f