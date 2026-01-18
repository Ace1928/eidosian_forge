import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
class TestSytrd:

    @pytest.mark.parametrize('dtype', REAL_DTYPES)
    def test_sytrd_with_zero_dim_array(self, dtype):
        A = np.zeros((0, 0), dtype=dtype)
        sytrd = get_lapack_funcs('sytrd', (A,))
        assert_raises(ValueError, sytrd, A)

    @pytest.mark.parametrize('dtype', REAL_DTYPES)
    @pytest.mark.parametrize('n', (1, 3))
    def test_sytrd(self, dtype, n):
        A = np.zeros((n, n), dtype=dtype)
        sytrd, sytrd_lwork = get_lapack_funcs(('sytrd', 'sytrd_lwork'), (A,))
        A[np.triu_indices_from(A)] = np.arange(1, n * (n + 1) // 2 + 1, dtype=dtype)
        lwork, info = sytrd_lwork(n)
        assert_equal(info, 0)
        data, d, e, tau, info = sytrd(A, lower=1, lwork=lwork)
        assert_equal(info, 0)
        assert_allclose(data, A, atol=5 * np.finfo(dtype).eps, rtol=1.0)
        assert_allclose(d, np.diag(A))
        assert_allclose(e, 0.0)
        assert_allclose(tau, 0.0)
        data, d, e, tau, info = sytrd(A, lwork=lwork)
        assert_equal(info, 0)
        T = np.zeros_like(A, dtype=dtype)
        k = np.arange(A.shape[0])
        T[k, k] = d
        k2 = np.arange(A.shape[0] - 1)
        T[k2 + 1, k2] = e
        T[k2, k2 + 1] = e
        Q = np.eye(n, n, dtype=dtype)
        for i in range(n - 1):
            v = np.zeros(n, dtype=dtype)
            v[:i] = data[:i, i + 1]
            v[i] = 1.0
            H = np.eye(n, n, dtype=dtype) - tau[i] * np.outer(v, v)
            Q = np.dot(H, Q)
        i_lower = np.tril_indices(n, -1)
        A[i_lower] = A.T[i_lower]
        QTAQ = np.dot(Q.T, np.dot(A, Q))
        assert_allclose(QTAQ, T, atol=5 * np.finfo(dtype).eps, rtol=1.0)