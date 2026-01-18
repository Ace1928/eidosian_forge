import functools
import warnings
import operator
import types
import numpy as np
from . import numeric as _nx
from .numeric import result_type, NaN, asanyarray, ndim
from numpy.core.multiarray import add_docstring
from numpy.core import overrides
def _geomspace_dispatcher(start, stop, num=None, endpoint=None, dtype=None, axis=None):
    return (start, stop)