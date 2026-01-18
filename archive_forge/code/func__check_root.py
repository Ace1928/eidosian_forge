from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def _check_root(self, f, method, f_tol=0.01):
    if method == 'krylov':
        for jac_method in ['gmres', 'bicgstab', 'cgs', 'minres', 'tfqmr']:
            if jac_method in f.ROOT_JAC_KSP_BAD:
                continue
            res = root(f, f.xin, method=method, options={'ftol': f_tol, 'maxiter': 200, 'disp': 0, 'jac_options': {'method': jac_method}})
            assert_(np.absolute(res.fun).max() < f_tol)
    res = root(f, f.xin, method=method, options={'ftol': f_tol, 'maxiter': 200, 'disp': 0})
    assert_(np.absolute(res.fun).max() < f_tol)