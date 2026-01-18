import math
import numpy as np
from numpy import array, eye, exp, random
from numpy.testing import (
from scipy.sparse import csc_matrix, csc_array, SparseEfficiencyWarning
from scipy.sparse._construct import eye as speye
from scipy.sparse.linalg._matfuncs import (expm, _expm,
from scipy.sparse._sputils import matrix
from scipy.linalg import logm
from scipy.special import factorial, binom
import scipy.sparse
import scipy.sparse.linalg
def _burkardt_13_power(n, p):
    """
    A helper function for testing matrix functions.

    Parameters
    ----------
    n : integer greater than 1
        Order of the square matrix to be returned.
    p : non-negative integer
        Power of the matrix.

    Returns
    -------
    out : ndarray representing a square matrix
        A Forsythe matrix of order n, raised to the power p.

    """
    if n != int(n) or n < 2:
        raise ValueError('n must be an integer greater than 1')
    n = int(n)
    if p != int(p) or p < 0:
        raise ValueError('p must be a non-negative integer')
    p = int(p)
    a, b = divmod(p, n)
    large = np.power(10.0, -n * a)
    small = large * np.power(10.0, -n)
    return np.diag([large] * (n - b), b) + np.diag([small] * b, b - n)