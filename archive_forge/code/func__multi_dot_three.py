import functools
import operator
import warnings
from typing import NamedTuple, Any
from .._utils import set_module
from numpy.core import (
from numpy.core.multiarray import normalize_axis_index
from numpy.core import overrides
from numpy.lib.twodim_base import triu, eye
from numpy.linalg import _umath_linalg
from numpy._typing import NDArray
def _multi_dot_three(A, B, C, out=None):
    """
    Find the best order for three arrays and do the multiplication.

    For three arguments `_multi_dot_three` is approximately 15 times faster
    than `_multi_dot_matrix_chain_order`

    """
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    cost1 = a0 * b1c0 * (a1b0 + c1)
    cost2 = a1b0 * c1 * (a0 + b1c0)
    if cost1 < cost2:
        return dot(dot(A, B), C, out=out)
    else:
        return dot(A, dot(B, C), out=out)