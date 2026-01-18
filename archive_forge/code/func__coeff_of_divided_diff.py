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
def _coeff_of_divided_diff(x):
    """
    Returns the coefficients of the divided difference.

    Parameters
    ----------
    x : array, shape (n,)
        Array which is used for the computation of divided difference.

    Returns
    -------
    res : array_like, shape (n,)
        Coefficients of the divided difference.

    Notes
    -----
    Vector ``x`` should have unique elements, otherwise an error division by
    zero might be raised.

    No checks are performed.

    """
    n = x.shape[0]
    res = np.zeros(n)
    for i in range(n):
        pp = 1.0
        for k in range(n):
            if k != i:
                pp *= x[i] - x[k]
        res[i] = 1.0 / pp
    return res