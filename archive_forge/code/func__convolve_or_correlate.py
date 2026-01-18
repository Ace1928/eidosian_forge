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
def _convolve_or_correlate(f, a, v, mode, propagate_mask):
    """
    Helper function for ma.correlate and ma.convolve
    """
    if propagate_mask:
        mask = f(getmaskarray(a), np.ones(np.shape(v), dtype=bool), mode=mode) | f(np.ones(np.shape(a), dtype=bool), getmaskarray(v), mode=mode)
        data = f(getdata(a), getdata(v), mode=mode)
    else:
        mask = ~f(~getmaskarray(a), ~getmaskarray(v))
        data = f(filled(a, 0), filled(v, 0), mode=mode)
    return masked_array(data, mask=mask)