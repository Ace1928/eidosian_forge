from itertools import product
import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import issparse, lil_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import least_squares, Bounds
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
from scipy.optimize._lsq.common import EPS, make_strictly_feasible, CL_scaling_vector
def _jac(self, x):
    J = lil_matrix((self.n, self.n))
    i = np.arange(self.n)
    J[i, i] = 3 - 2 * x
    i = np.arange(1, self.n)
    J[i, i - 1] = -1
    i = np.arange(self.n - 1)
    J[i, i + 1] = -2
    return J