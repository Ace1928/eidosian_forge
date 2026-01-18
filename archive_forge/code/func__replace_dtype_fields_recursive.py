import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
def _replace_dtype_fields_recursive(dtype, primitive_dtype):
    """Private function allowing recursion in _replace_dtype_fields."""
    _recurse = _replace_dtype_fields_recursive
    if dtype.names is not None:
        descr = []
        for name in dtype.names:
            field = dtype.fields[name]
            if len(field) == 3:
                name = (field[-1], name)
            descr.append((name, _recurse(field[0], primitive_dtype)))
        new_dtype = np.dtype(descr)
    elif dtype.subdtype:
        descr = list(dtype.subdtype)
        descr[0] = _recurse(dtype.subdtype[0], primitive_dtype)
        new_dtype = np.dtype(tuple(descr))
    else:
        new_dtype = primitive_dtype
    if new_dtype == dtype:
        new_dtype = dtype
    return new_dtype