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
def _mask_propagate(a, axis):
    """
    Mask whole 1-d vectors of an array that contain masked values.
    """
    a = array(a, subok=False)
    m = getmask(a)
    if m is nomask or not m.any() or axis is None:
        return a
    a._mask = a._mask.copy()
    axes = normalize_axis_tuple(axis, a.ndim)
    for ax in axes:
        a._mask |= m.any(axis=ax, keepdims=True)
    return a