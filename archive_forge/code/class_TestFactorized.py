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
class TestFactorized:

    def setup_method(self):
        n = 5
        d = arange(n) + 1
        self.n = n
        self.A = spdiags((d, 2 * d, d[::-1]), (-3, 0, 5), n, n).tocsc()
        random.seed(1234)

    def _check_singular(self):
        A = csc_matrix((5, 5), dtype='d')
        b = ones(5)
        assert_array_almost_equal(0.0 * b, factorized(A)(b))

    def _check_non_singular(self):
        n = 5
        a = csc_matrix(random.rand(n, n))
        b = ones(n)
        expected = splu(a).solve(b)
        assert_array_almost_equal(factorized(a)(b), expected)

    def test_singular_without_umfpack(self):
        use_solver(useUmfpack=False)
        with assert_raises(RuntimeError, match='Factor is exactly singular'):
            self._check_singular()

    @pytest.mark.skipif(not has_umfpack, reason='umfpack not available')
    def test_singular_with_umfpack(self):
        use_solver(useUmfpack=True)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'divide by zero encountered in double_scalars')
            assert_warns(umfpack.UmfpackWarning, self._check_singular)

    def test_non_singular_without_umfpack(self):
        use_solver(useUmfpack=False)
        self._check_non_singular()

    @pytest.mark.skipif(not has_umfpack, reason='umfpack not available')
    def test_non_singular_with_umfpack(self):
        use_solver(useUmfpack=True)
        self._check_non_singular()

    def test_cannot_factorize_nonsquare_matrix_without_umfpack(self):
        use_solver(useUmfpack=False)
        msg = 'can only factor square matrices'
        with assert_raises(ValueError, match=msg):
            factorized(self.A[:, :4])

    @pytest.mark.skipif(not has_umfpack, reason='umfpack not available')
    def test_factorizes_nonsquare_matrix_with_umfpack(self):
        use_solver(useUmfpack=True)
        factorized(self.A[:, :4])

    def test_call_with_incorrectly_sized_matrix_without_umfpack(self):
        use_solver(useUmfpack=False)
        solve = factorized(self.A)
        b = random.rand(4)
        B = random.rand(4, 3)
        BB = random.rand(self.n, 3, 9)
        with assert_raises(ValueError, match='is of incompatible size'):
            solve(b)
        with assert_raises(ValueError, match='is of incompatible size'):
            solve(B)
        with assert_raises(ValueError, match='object too deep for desired array'):
            solve(BB)

    @pytest.mark.skipif(not has_umfpack, reason='umfpack not available')
    def test_call_with_incorrectly_sized_matrix_with_umfpack(self):
        use_solver(useUmfpack=True)
        solve = factorized(self.A)
        b = random.rand(4)
        B = random.rand(4, 3)
        BB = random.rand(self.n, 3, 9)
        solve(b)
        msg = 'object too deep for desired array'
        with assert_raises(ValueError, match=msg):
            solve(B)
        with assert_raises(ValueError, match=msg):
            solve(BB)

    def test_call_with_cast_to_complex_without_umfpack(self):
        use_solver(useUmfpack=False)
        solve = factorized(self.A)
        b = random.rand(4)
        for t in [np.complex64, np.complex128]:
            with assert_raises(TypeError, match='Cannot cast array data'):
                solve(b.astype(t))

    @pytest.mark.skipif(not has_umfpack, reason='umfpack not available')
    def test_call_with_cast_to_complex_with_umfpack(self):
        use_solver(useUmfpack=True)
        solve = factorized(self.A)
        b = random.rand(4)
        for t in [np.complex64, np.complex128]:
            assert_warns(ComplexWarning, solve, b.astype(t))

    @pytest.mark.skipif(not has_umfpack, reason='umfpack not available')
    def test_assume_sorted_indices_flag(self):
        unsorted_inds = np.array([2, 0, 1, 0])
        data = np.array([10, 16, 5, 0.4])
        indptr = np.array([0, 1, 2, 4])
        A = csc_matrix((data, unsorted_inds, indptr), (3, 3))
        b = ones(3)
        use_solver(useUmfpack=True, assumeSortedIndices=True)
        with assert_raises(RuntimeError, match='UMFPACK_ERROR_invalid_matrix'):
            factorized(A)
        use_solver(useUmfpack=True, assumeSortedIndices=False)
        expected = splu(A.copy()).solve(b)
        assert_equal(A.has_sorted_indices, 0)
        assert_array_almost_equal(factorized(A)(b), expected)

    @pytest.mark.slow
    @pytest.mark.skipif(not has_umfpack, reason='umfpack not available')
    def test_bug_8278(self):
        check_free_memory(8000)
        use_solver(useUmfpack=True)
        A, b = setup_bug_8278()
        A = A.tocsc()
        f = factorized(A)
        x = f(b)
        assert_array_almost_equal(A @ x, b)