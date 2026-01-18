import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
class TestSchur:

    def check_schur(self, a, t, u, rtol, atol):
        assert_allclose(u @ t @ u.conj().T, a, rtol=rtol, atol=atol, err_msg="Schur decomposition does not match 'a'")
        assert_allclose(u @ u.conj().T - np.eye(len(u)), 0, rtol=0, atol=atol, err_msg='u is not unitary')

    def test_simple(self):
        a = [[8, 12, 3], [2, 9, 3], [10, 3, 6]]
        t, z = schur(a)
        self.check_schur(a, t, z, rtol=1e-14, atol=5e-15)
        tc, zc = schur(a, 'complex')
        assert_(np.any(ravel(iscomplex(zc))) and np.any(ravel(iscomplex(tc))))
        self.check_schur(a, tc, zc, rtol=1e-14, atol=5e-15)
        tc2, zc2 = rsf2csf(tc, zc)
        self.check_schur(a, tc2, zc2, rtol=1e-14, atol=5e-15)

    @pytest.mark.parametrize('sort, expected_diag', [('lhp', [-np.sqrt(2), -0.5, np.sqrt(2), 0.5]), ('rhp', [np.sqrt(2), 0.5, -np.sqrt(2), -0.5]), ('iuc', [-0.5, 0.5, np.sqrt(2), -np.sqrt(2)]), ('ouc', [np.sqrt(2), -np.sqrt(2), -0.5, 0.5]), (lambda x: x >= 0.0, [np.sqrt(2), 0.5, -np.sqrt(2), -0.5])])
    def test_sort(self, sort, expected_diag):
        a = [[4.0, 3.0, 1.0, -1.0], [-4.5, -3.5, -1.0, 1.0], [9.0, 6.0, -4.0, 4.5], [6.0, 4.0, -3.0, 3.5]]
        t, u, sdim = schur(a, sort=sort)
        self.check_schur(a, t, u, rtol=1e-14, atol=5e-15)
        assert_allclose(np.diag(t), expected_diag, rtol=1e-12)
        assert_equal(2, sdim)

    def test_sort_errors(self):
        a = [[4.0, 3.0, 1.0, -1.0], [-4.5, -3.5, -1.0, 1.0], [9.0, 6.0, -4.0, 4.5], [6.0, 4.0, -3.0, 3.5]]
        assert_raises(ValueError, schur, a, sort='unsupported')
        assert_raises(ValueError, schur, a, sort=1)

    def test_check_finite(self):
        a = [[8, 12, 3], [2, 9, 3], [10, 3, 6]]
        t, z = schur(a, check_finite=False)
        assert_array_almost_equal(z @ t @ z.conj().T, a)