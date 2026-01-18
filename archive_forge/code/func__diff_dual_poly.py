import operator
from math import prod
import numpy as np
from scipy._lib._util import normalize_axis_index
from scipy.linalg import (get_lapack_funcs, LinAlgError,
from scipy.optimize import minimize_scalar
from . import _bspl
from . import _fitpack_impl
from scipy.sparse import csr_array
from scipy.special import poch
from itertools import combinations
def _diff_dual_poly(j, k, y, d, t):
    """
    d-th derivative of the dual polynomial $p_{j,k}(y)$
    """
    if d == 0:
        return _dual_poly(j, k, t, y)
    if d == k:
        return poch(1, k)
    comb = list(combinations(range(j + 1, j + k + 1), d))
    res = 0
    for i in range(len(comb) * len(comb[0])):
        res += np.prod([y - t[j + p] for p in range(1, k + 1) if j + p not in comb[i // d]])
    return res