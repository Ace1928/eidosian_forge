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
def _delegate_binop(self, other):
    if isinstance(other, type(self)):
        return False
    array_ufunc = getattr(other, '__array_ufunc__', False)
    if array_ufunc is False:
        other_priority = getattr(other, '__array_priority__', -1000000)
        return self.__array_priority__ < other_priority
    else:
        return array_ufunc is None