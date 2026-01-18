import collections.abc
import functools
import re
import sys
import warnings
from .._utils import set_module
import numpy as np
import numpy.core.numeric as _nx
from numpy.core import transpose
from numpy.core.numeric import (
from numpy.core.umath import (
from numpy.core.fromnumeric import (
from numpy.core.numerictypes import typecodes
from numpy.core import overrides
from numpy.core.function_base import add_newdoc
from numpy.lib.twodim_base import diag
from numpy.core.multiarray import (
from numpy.core.umath import _add_newdoc_ufunc as add_newdoc_ufunc
import builtins
from numpy.lib.histograms import histogram, histogramdd  # noqa: F401
def _quantile_ureduce_func(a: np.array, q: np.array, axis: int=None, out=None, overwrite_input: bool=False, method='linear') -> np.array:
    if q.ndim > 2:
        raise ValueError('q must be a scalar or 1d')
    if overwrite_input:
        if axis is None:
            axis = 0
            arr = a.ravel()
        else:
            arr = a
    elif axis is None:
        axis = 0
        arr = a.flatten()
    else:
        arr = a.copy()
    result = _quantile(arr, quantiles=q, axis=axis, method=method, out=out)
    return result