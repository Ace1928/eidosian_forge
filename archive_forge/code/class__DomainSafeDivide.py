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
class _DomainSafeDivide:
    """
    Define a domain for safe division.

    """

    def __init__(self, tolerance=None):
        self.tolerance = tolerance

    def __call__(self, a, b):
        if self.tolerance is None:
            self.tolerance = np.finfo(float).tiny
        a, b = (np.asarray(a), np.asarray(b))
        with np.errstate(invalid='ignore'):
            return umath.absolute(a) * self.tolerance >= umath.absolute(b)