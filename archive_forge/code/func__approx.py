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
def _approx(n):
    val = np.power(2, n / 2) * gamma(n / 2 + 1)
    mask = np.ones_like(n, dtype=np.float64)
    mask[n % 2 == 1] = sqrt(2 / pi)
    return val * mask