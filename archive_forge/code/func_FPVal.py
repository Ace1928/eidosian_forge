from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def FPVal(sig, exp=None, fps=None, ctx=None):
    """Return a floating-point value of value `val` and sort `fps`.
    If `ctx=None`, then the global context is used.

    >>> v = FPVal(20.0, FPSort(8, 24))
    >>> v
    1.25*(2**4)
    >>> print("0x%.8x" % v.exponent_as_long(False))
    0x00000004
    >>> v = FPVal(2.25, FPSort(8, 24))
    >>> v
    1.125*(2**1)
    >>> v = FPVal(-2.25, FPSort(8, 24))
    >>> v
    -1.125*(2**1)
    >>> FPVal(-0.0, FPSort(8, 24))
    -0.0
    >>> FPVal(0.0, FPSort(8, 24))
    +0.0
    >>> FPVal(+0.0, FPSort(8, 24))
    +0.0
    """
    ctx = _get_ctx(ctx)
    if is_fp_sort(exp):
        fps = exp
        exp = None
    elif fps is None:
        fps = _dflt_fps(ctx)
    _z3_assert(is_fp_sort(fps), 'sort mismatch')
    if exp is None:
        exp = 0
    val = _to_float_str(sig)
    if val == 'NaN' or val == 'nan':
        return fpNaN(fps)
    elif val == '-0.0':
        return fpMinusZero(fps)
    elif val == '0.0' or val == '+0.0':
        return fpPlusZero(fps)
    elif val == '+oo' or val == '+inf' or val == '+Inf':
        return fpPlusInfinity(fps)
    elif val == '-oo' or val == '-inf' or val == '-Inf':
        return fpMinusInfinity(fps)
    else:
        return FPNumRef(Z3_mk_numeral(ctx.ref(), val, fps.ast), ctx)