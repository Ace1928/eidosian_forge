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
def _get_dtype_of(obj):
    """ Convert the argument for *_fill_value into a dtype """
    if isinstance(obj, np.dtype):
        return obj
    elif hasattr(obj, 'dtype'):
        return obj.dtype
    else:
        return np.asanyarray(obj).dtype