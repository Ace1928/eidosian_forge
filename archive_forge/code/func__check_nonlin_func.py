from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def _check_nonlin_func(self, f, func, f_tol=0.01):
    if func == SOLVERS['krylov']:
        for method in ['gmres', 'bicgstab', 'cgs', 'minres', 'tfqmr']:
            if method in f.JAC_KSP_BAD:
                continue
            x = func(f, f.xin, method=method, line_search=None, f_tol=f_tol, maxiter=200, verbose=0)
            assert_(np.absolute(f(x)).max() < f_tol)
    x = func(f, f.xin, f_tol=f_tol, maxiter=200, verbose=0)
    assert_(np.absolute(f(x)).max() < f_tol)