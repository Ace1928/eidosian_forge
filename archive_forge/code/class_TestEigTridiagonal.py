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
class TestEigTridiagonal:

    def setup_method(self):
        self.create_trimat()

    def create_trimat(self):
        """Create the full matrix `self.fullmat`, `self.d`, and `self.e`."""
        N = 10
        self.d = full(N, 1.0)
        self.e = full(N - 1, -1.0)
        self.full_mat = diag(self.d) + diag(self.e, -1) + diag(self.e, 1)
        ew, ev = linalg.eig(self.full_mat)
        ew = ew.real
        args = argsort(ew)
        self.w = ew[args]
        self.evec = ev[:, args]

    def test_degenerate(self):
        """Test error conditions."""
        assert_raises(ValueError, eigvalsh_tridiagonal, self.d, self.e[:-1])
        assert_raises(TypeError, eigvalsh_tridiagonal, self.d, self.e * 1j)
        assert_raises(TypeError, eigvalsh_tridiagonal, self.d, self.e, lapack_driver=1.0)
        assert_raises(ValueError, eigvalsh_tridiagonal, self.d, self.e, lapack_driver='foo')
        assert_raises(ValueError, eigvalsh_tridiagonal, self.d, self.e, select='i', select_range=(0, -1))

    def test_eigvalsh_tridiagonal(self):
        """Compare eigenvalues of eigvalsh_tridiagonal with those of eig."""
        for driver in ('sterf', 'stev', 'stebz', 'stemr', 'auto'):
            w = eigvalsh_tridiagonal(self.d, self.e, lapack_driver=driver)
            assert_array_almost_equal(sort(w), self.w)
        for driver in ('sterf', 'stev'):
            assert_raises(ValueError, eigvalsh_tridiagonal, self.d, self.e, lapack_driver='stev', select='i', select_range=(0, 1))
        for driver in ('stebz', 'stemr', 'auto'):
            w_ind = eigvalsh_tridiagonal(self.d, self.e, select='i', select_range=(0, len(self.d) - 1), lapack_driver=driver)
            assert_array_almost_equal(sort(w_ind), self.w)
            ind1 = 2
            ind2 = 6
            w_ind = eigvalsh_tridiagonal(self.d, self.e, select='i', select_range=(ind1, ind2), lapack_driver=driver)
            assert_array_almost_equal(sort(w_ind), self.w[ind1:ind2 + 1])
            v_lower = self.w[ind1] - 1e-05
            v_upper = self.w[ind2] + 1e-05
            w_val = eigvalsh_tridiagonal(self.d, self.e, select='v', select_range=(v_lower, v_upper), lapack_driver=driver)
            assert_array_almost_equal(sort(w_val), self.w[ind1:ind2 + 1])

    def test_eigh_tridiagonal(self):
        """Compare eigenvalues and eigenvectors of eigh_tridiagonal
           with those of eig. """
        assert_raises(ValueError, eigh_tridiagonal, self.d, self.e, lapack_driver='sterf')
        for driver in ('stebz', 'stev', 'stemr', 'auto'):
            w, evec = eigh_tridiagonal(self.d, self.e, lapack_driver=driver)
            evec_ = evec[:, argsort(w)]
            assert_array_almost_equal(sort(w), self.w)
            assert_array_almost_equal(abs(evec_), abs(self.evec))
        assert_raises(ValueError, eigh_tridiagonal, self.d, self.e, lapack_driver='stev', select='i', select_range=(0, 1))
        for driver in ('stebz', 'stemr', 'auto'):
            ind1 = 0
            ind2 = len(self.d) - 1
            w, evec = eigh_tridiagonal(self.d, self.e, select='i', select_range=(ind1, ind2), lapack_driver=driver)
            assert_array_almost_equal(sort(w), self.w)
            assert_array_almost_equal(abs(evec), abs(self.evec))
            ind1 = 2
            ind2 = 6
            w, evec = eigh_tridiagonal(self.d, self.e, select='i', select_range=(ind1, ind2), lapack_driver=driver)
            assert_array_almost_equal(sort(w), self.w[ind1:ind2 + 1])
            assert_array_almost_equal(abs(evec), abs(self.evec[:, ind1:ind2 + 1]))
            v_lower = self.w[ind1] - 1e-05
            v_upper = self.w[ind2] + 1e-05
            w, evec = eigh_tridiagonal(self.d, self.e, select='v', select_range=(v_lower, v_upper), lapack_driver=driver)
            assert_array_almost_equal(sort(w), self.w[ind1:ind2 + 1])
            assert_array_almost_equal(abs(evec), abs(self.evec[:, ind1:ind2 + 1]))