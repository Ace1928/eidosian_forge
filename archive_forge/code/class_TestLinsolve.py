import sys
import threading
import numpy as np
from numpy import array, finfo, arange, eye, all, unique, ones, dot
import numpy.random as random
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
import scipy.linalg
from scipy.linalg import norm, inv
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
from scipy.sparse.linalg import SuperLU
from scipy.sparse.linalg._dsolve import (spsolve, use_solver, splu, spilu,
import scipy.sparse
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import ComplexWarning
class TestLinsolve:

    def setup_method(self):
        use_solver(useUmfpack=False)

    def test_singular(self):
        A = csc_matrix((5, 5), dtype='d')
        b = array([1, 2, 3, 4, 5], dtype='d')
        with suppress_warnings() as sup:
            sup.filter(MatrixRankWarning, 'Matrix is exactly singular')
            x = spsolve(A, b)
        assert_(not np.isfinite(x).any())

    def test_singular_gh_3312(self):
        ij = np.array([(17, 0), (17, 6), (17, 12), (10, 13)], dtype=np.int32)
        v = np.array([0.284213, 0.94933781, 0.15767017, 0.38797296])
        A = csc_matrix((v, ij.T), shape=(20, 20))
        b = np.arange(20)
        try:
            with suppress_warnings() as sup:
                sup.filter(MatrixRankWarning, 'Matrix is exactly singular')
                x = spsolve(A, b)
            assert not np.isfinite(x).any()
        except RuntimeError:
            pass

    @pytest.mark.parametrize('format', ['csc', 'csr'])
    @pytest.mark.parametrize('idx_dtype', [np.int32, np.int64])
    def test_twodiags(self, format: str, idx_dtype: np.dtype):
        A = spdiags([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]], [0, 1], 5, 5, format=format)
        b = array([1, 2, 3, 4, 5])
        cond_A = norm(A.toarray(), 2) * norm(inv(A.toarray()), 2)
        for t in ['f', 'd', 'F', 'D']:
            eps = finfo(t).eps
            b = b.astype(t)
            Asp = A.astype(t)
            Asp.indices = Asp.indices.astype(idx_dtype, copy=False)
            Asp.indptr = Asp.indptr.astype(idx_dtype, copy=False)
            x = spsolve(Asp, b)
            assert_(norm(b - Asp @ x) < 10 * cond_A * eps)

    def test_bvector_smoketest(self):
        Adense = array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        As = csc_matrix(Adense)
        random.seed(1234)
        x = random.randn(3)
        b = As @ x
        x2 = spsolve(As, b)
        assert_array_almost_equal(x, x2)

    def test_bmatrix_smoketest(self):
        Adense = array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        As = csc_matrix(Adense)
        random.seed(1234)
        x = random.randn(3, 4)
        Bdense = As.dot(x)
        Bs = csc_matrix(Bdense)
        x2 = spsolve(As, Bs)
        assert_array_almost_equal(x, x2.toarray())

    @sup_sparse_efficiency
    def test_non_square(self):
        A = ones((3, 4))
        b = ones((4, 1))
        assert_raises(ValueError, spsolve, A, b)
        A2 = csc_matrix(eye(3))
        b2 = array([1.0, 2.0])
        assert_raises(ValueError, spsolve, A2, b2)

    @sup_sparse_efficiency
    def test_example_comparison(self):
        row = array([0, 0, 1, 2, 2, 2])
        col = array([0, 2, 2, 0, 1, 2])
        data = array([1, 2, 3, -4, 5, 6])
        sM = csr_matrix((data, (row, col)), shape=(3, 3), dtype=float)
        M = sM.toarray()
        row = array([0, 0, 1, 1, 0, 0])
        col = array([0, 2, 1, 1, 0, 0])
        data = array([1, 1, 1, 1, 1, 1])
        sN = csr_matrix((data, (row, col)), shape=(3, 3), dtype=float)
        N = sN.toarray()
        sX = spsolve(sM, sN)
        X = scipy.linalg.solve(M, N)
        assert_array_almost_equal(X, sX.toarray())

    @sup_sparse_efficiency
    @pytest.mark.skipif(not has_umfpack, reason='umfpack not available')
    def test_shape_compatibility(self):
        use_solver(useUmfpack=True)
        A = csc_matrix([[1.0, 0], [0, 2]])
        bs = [[1, 6], array([1, 6]), [[1], [6]], array([[1], [6]]), csc_matrix([[1], [6]]), csr_matrix([[1], [6]]), dok_matrix([[1], [6]]), bsr_matrix([[1], [6]]), array([[1.0, 2.0, 3.0], [6.0, 8.0, 10.0]]), csc_matrix([[1.0, 2.0, 3.0], [6.0, 8.0, 10.0]]), csr_matrix([[1.0, 2.0, 3.0], [6.0, 8.0, 10.0]]), dok_matrix([[1.0, 2.0, 3.0], [6.0, 8.0, 10.0]]), bsr_matrix([[1.0, 2.0, 3.0], [6.0, 8.0, 10.0]])]
        for b in bs:
            x = np.linalg.solve(A.toarray(), toarray(b))
            for spmattype in [csc_matrix, csr_matrix, dok_matrix, lil_matrix]:
                x1 = spsolve(spmattype(A), b, use_umfpack=True)
                x2 = spsolve(spmattype(A), b, use_umfpack=False)
                if x.ndim == 2 and x.shape[1] == 1:
                    x = x.ravel()
                assert_array_almost_equal(toarray(x1), x, err_msg=repr((b, spmattype, 1)))
                assert_array_almost_equal(toarray(x2), x, err_msg=repr((b, spmattype, 2)))
                if issparse(b) and x.ndim > 1:
                    assert_(issparse(x1), repr((b, spmattype, 1)))
                    assert_(issparse(x2), repr((b, spmattype, 2)))
                else:
                    assert_(isinstance(x1, np.ndarray), repr((b, spmattype, 1)))
                    assert_(isinstance(x2, np.ndarray), repr((b, spmattype, 2)))
                if x.ndim == 1:
                    assert_equal(x1.shape, (A.shape[1],))
                    assert_equal(x2.shape, (A.shape[1],))
                else:
                    assert_equal(x1.shape, x.shape)
                    assert_equal(x2.shape, x.shape)
        A = csc_matrix((3, 3))
        b = csc_matrix((1, 3))
        assert_raises(ValueError, spsolve, A, b)

    @sup_sparse_efficiency
    def test_ndarray_support(self):
        A = array([[1.0, 2.0], [2.0, 0.0]])
        x = array([[1.0, 1.0], [0.5, -0.5]])
        b = array([[2.0, 0.0], [2.0, 2.0]])
        assert_array_almost_equal(x, spsolve(A, b))

    def test_gssv_badinput(self):
        N = 10
        d = arange(N) + 1.0
        A = spdiags((d, 2 * d, d[::-1]), (-3, 0, 5), N, N)
        for spmatrix in (csc_matrix, csr_matrix):
            A = spmatrix(A)
            b = np.arange(N)

            def not_c_contig(x):
                return x.repeat(2)[::2]

            def not_1dim(x):
                return x[:, None]

            def bad_type(x):
                return x.astype(bool)

            def too_short(x):
                return x[:-1]
            badops = [not_c_contig, not_1dim, bad_type, too_short]
            for badop in badops:
                msg = f'{spmatrix!r} {badop!r}'
                assert_raises((ValueError, TypeError), _superlu.gssv, N, A.nnz, badop(A.data), A.indices, A.indptr, b, int(spmatrix == csc_matrix), err_msg=msg)
                assert_raises((ValueError, TypeError), _superlu.gssv, N, A.nnz, A.data, badop(A.indices), A.indptr, b, int(spmatrix == csc_matrix), err_msg=msg)
                assert_raises((ValueError, TypeError), _superlu.gssv, N, A.nnz, A.data, A.indices, badop(A.indptr), b, int(spmatrix == csc_matrix), err_msg=msg)

    def test_sparsity_preservation(self):
        ident = csc_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        b = csc_matrix([[0, 1], [1, 0], [0, 0]])
        x = spsolve(ident, b)
        assert_equal(ident.nnz, 3)
        assert_equal(b.nnz, 2)
        assert_equal(x.nnz, 2)
        assert_allclose(x.A, b.A, atol=1e-12, rtol=1e-12)

    def test_dtype_cast(self):
        A_real = scipy.sparse.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        A_complex = scipy.sparse.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5 + 1j]])
        b_real = np.array([1, 1, 1])
        b_complex = np.array([1, 1, 1]) + 1j * np.array([1, 1, 1])
        x = spsolve(A_real, b_real)
        assert_(np.issubdtype(x.dtype, np.floating))
        x = spsolve(A_real, b_complex)
        assert_(np.issubdtype(x.dtype, np.complexfloating))
        x = spsolve(A_complex, b_real)
        assert_(np.issubdtype(x.dtype, np.complexfloating))
        x = spsolve(A_complex, b_complex)
        assert_(np.issubdtype(x.dtype, np.complexfloating))

    @pytest.mark.slow
    @pytest.mark.skipif(not has_umfpack, reason='umfpack not available')
    def test_bug_8278(self):
        check_free_memory(8000)
        use_solver(useUmfpack=True)
        A, b = setup_bug_8278()
        x = spsolve(A, b)
        assert_array_almost_equal(A @ x, b)