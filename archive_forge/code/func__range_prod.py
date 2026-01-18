import operator
import numpy as np
import math
import warnings
from collections import defaultdict
from heapq import heapify, heappop
from numpy import (pi, asarray, floor, isscalar, iscomplex, real,
from . import _ufuncs
from ._ufuncs import (mathieu_a, mathieu_b, iv, jv, gamma,
from . import _specfun
from ._comb import _comb_int
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def _range_prod(lo, hi, k=1):
    """
    Product of a range of numbers spaced k apart (from hi).

    For k=1, this returns the product of
    lo * (lo+1) * (lo+2) * ... * (hi-2) * (hi-1) * hi
    = hi! / (lo-1)!

    For k>1, it correspond to taking only every k'th number when
    counting down from hi - e.g. 18!!!! = _range_prod(1, 18, 4).

    Breaks into smaller products first for speed:
    _range_prod(2, 9) = ((2*3)*(4*5))*((6*7)*(8*9))
    """
    if lo + k < hi:
        mid = (hi + lo) // 2
        if k > 1:
            mid = mid - (mid - hi) % k
        return _range_prod(lo, mid, k) * _range_prod(mid + k, hi, k)
    elif lo + k == hi:
        return lo * hi
    else:
        return hi