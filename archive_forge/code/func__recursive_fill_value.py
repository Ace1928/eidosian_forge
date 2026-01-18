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
def _recursive_fill_value(dtype, f):
    """
    Recursively produce a fill value for `dtype`, calling f on scalar dtypes
    """
    if dtype.names is not None:
        vals = tuple((np.array(_recursive_fill_value(dtype[name], f)) for name in dtype.names))
        return np.array(vals, dtype=dtype)[()]
    elif dtype.subdtype:
        subtype, shape = dtype.subdtype
        subval = _recursive_fill_value(subtype, f)
        return np.full(shape, subval)
    else:
        return f(dtype)