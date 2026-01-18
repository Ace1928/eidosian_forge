import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _inverse_poly_zero(a, b, c, d, fa, fb, fc, fd):
    """Inverse cubic interpolation f-values -> x-values

    Given four points (fa, a), (fb, b), (fc, c), (fd, d) with
    fa, fb, fc, fd all distinct, find poly IP(y) through the 4 points
    and compute x=IP(0).
    """
    return _interpolated_poly([fa, fb, fc, fd], [a, b, c, d], 0)