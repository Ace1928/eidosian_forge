import functools
import operator
import warnings
from typing import NamedTuple, Any
from .._utils import set_module
from numpy.core import (
from numpy.core.multiarray import normalize_axis_index
from numpy.core import overrides
from numpy.lib.twodim_base import triu, eye
from numpy.linalg import _umath_linalg
from numpy._typing import NDArray
def _convertarray(a):
    t, result_t = _commonType(a)
    a = a.astype(t).T.copy()
    return (a, t, result_t)