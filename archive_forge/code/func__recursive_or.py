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
def _recursive_or(a, b):
    """do a|=b on each field of a, recursively"""
    for name in a.dtype.names:
        af, bf = (a[name], b[name])
        if af.dtype.names is not None:
            _recursive_or(af, bf)
        else:
            af |= bf