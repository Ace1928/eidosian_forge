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
def _lerp(a, b, t, out=None):
    """
    Compute the linear interpolation weighted by gamma on each point of
    two same shape array.

    a : array_like
        Left bound.
    b : array_like
        Right bound.
    t : array_like
        The interpolation weight.
    out : array_like
        Output array.
    """
    diff_b_a = subtract(b, a)
    lerp_interpolation = asanyarray(add(a, diff_b_a * t, out=out))
    subtract(b, diff_b_a * (1 - t), out=lerp_interpolation, where=t >= 0.5, casting='unsafe', dtype=type(lerp_interpolation.dtype))
    if lerp_interpolation.ndim == 0 and out is None:
        lerp_interpolation = lerp_interpolation[()]
    return lerp_interpolation