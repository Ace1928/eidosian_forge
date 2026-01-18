from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def _check_autojac(self, A, b):

    def func(x):
        return A.dot(x) - b

    def jac(v):
        return A
    sol = nonlin.nonlin_solve(func, np.zeros(b.shape[0]), jac, maxiter=2, f_tol=1e-06, line_search=None, verbose=0)
    np.testing.assert_allclose(A @ sol, b, atol=1e-06)
    sol = nonlin.nonlin_solve(func, np.zeros(b.shape[0]), A, maxiter=2, f_tol=1e-06, line_search=None, verbose=0)
    np.testing.assert_allclose(A @ sol, b, atol=1e-06)