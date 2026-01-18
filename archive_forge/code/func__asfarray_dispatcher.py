import functools
from .._utils import set_module
import numpy.core.numeric as _nx
from numpy.core.numeric import asarray, asanyarray, isnan, zeros
from numpy.core import overrides, getlimits
from .ufunclike import isneginf, isposinf
def _asfarray_dispatcher(a, dtype=None):
    return (a,)