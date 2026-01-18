from __future__ import annotations
import operator
import numpy as np
from xarray.core import dtypes, duck_array_ops
def _call_possibly_missing_method(arg, name, args, kwargs):
    try:
        method = getattr(arg, name)
    except AttributeError:
        duck_array_ops.fail_on_dask_array_input(arg, func_name=name)
        if hasattr(arg, 'data'):
            duck_array_ops.fail_on_dask_array_input(arg.data, func_name=name)
        raise
    else:
        return method(*args, **kwargs)