import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def _is_conditionally_positive_definite(kernel, m):
    nx = 10
    ntests = 100
    for ndim in [1, 2, 3, 4, 5]:
        seq = Halton(ndim, scramble=False, seed=np.random.RandomState())
        for _ in range(ntests):
            x = 2 * seq.random(nx) - 1
            A = _rbfinterp_pythran._kernel_matrix(x, kernel)
            P = _vandermonde(x, m - 1)
            Q, R = np.linalg.qr(P, mode='complete')
            Q2 = Q[:, P.shape[1]:]
            B = Q2.T.dot(A).dot(Q2)
            try:
                np.linalg.cholesky(B)
            except np.linalg.LinAlgError:
                return False
    return True