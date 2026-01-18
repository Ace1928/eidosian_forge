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
def _dask_or_eager_func(name, eager_module=np, dask_module='dask.array'):
    """Create a function that dispatches to dask for dask array inputs."""

    def f(*args, **kwargs):
        if any((is_duck_dask_array(a) for a in args)):
            mod = import_module(dask_module) if isinstance(dask_module, str) else dask_module
            wrapped = getattr(mod, name)
        else:
            wrapped = getattr(eager_module, name)
        return wrapped(*args, **kwargs)
    return f