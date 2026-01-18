import functools
from .._utils import set_module
import numpy.core.numeric as _nx
from numpy.core.numeric import asarray, asanyarray, isnan, zeros
from numpy.core import overrides, getlimits
from .ufunclike import isneginf, isposinf
def _getmaxmin(t):
    from numpy.core import getlimits
    f = getlimits.finfo(t)
    return (f.max, f.min)