import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
class _TestCommon:
    """test common functionality shared by all sparse formats"""
    math_dtypes = supported_dtypes

    @classmethod
    def init_class(cls):
        cls.dat = array([[1, 0, 0, 2], [3, 0, 1, 0], [0, 2, 0, 0]], 'd')
        cls.datsp = cls.spcreator(cls.dat)
        cls.checked_dtypes = set(supported_dtypes).union(cls.math_dtypes)
        cls.dat_dtypes = {}
        cls.datsp_dtypes = {}
        for dtype in cls.checked_dtypes:
            cls.dat_dtypes[dtype] = cls.dat.astype(dtype)
            cls.datsp_dtypes[dtype] = cls.spcreator(cls.dat.astype(dtype))
        assert_equal(cls.dat, cls.dat_dtypes[np.float64])
        assert_equal(cls.datsp.toarray(), cls.datsp_dtypes[np.float64].toarray())

    def test_bool(self):

        def check(dtype):
            datsp = self.datsp_dtypes[dtype]
            assert_raises(ValueError, bool, datsp)
            assert_(self.spcreator([1]))
            assert_(not self.spcreator([0]))
        if isinstance(self, TestDOK):
            pytest.skip('Cannot create a rank <= 2 DOK matrix.')
        for dtype in self.checked_dtypes:
            check(dtype)

    def test_bool_rollover(self):
        dat = array([[True, False]])
        datsp = self.spcreator(dat)
        for _ in range(10):
            datsp = datsp + datsp
            dat = dat + dat
        assert_array_equal(dat, datsp.toarray())

    def test_eq(self):
        sup = suppress_warnings()
        sup.filter(SparseEfficiencyWarning)

        @sup
        @sup_complex
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            dat2 = dat.copy()
            dat2[:, 0] = 0
            datsp2 = self.spcreator(dat2)
            datbsr = bsr_matrix(dat)
            datcsr = csr_matrix(dat)
            datcsc = csc_matrix(dat)
            datlil = lil_matrix(dat)
            assert_array_equal_dtype(dat == dat2, (datsp == datsp2).toarray())
            assert_array_equal_dtype(dat == dat2, (datbsr == datsp2).toarray())
            assert_array_equal_dtype(dat == dat2, (datcsr == datsp2).toarray())
            assert_array_equal_dtype(dat == dat2, (datcsc == datsp2).toarray())
            assert_array_equal_dtype(dat == dat2, (datlil == datsp2).toarray())
            assert_array_equal_dtype(dat == datsp2, datsp2 == dat)
            assert_array_equal_dtype(dat == 0, (datsp == 0).toarray())
            assert_array_equal_dtype(dat == 1, (datsp == 1).toarray())
            assert_array_equal_dtype(dat == np.nan, (datsp == np.nan).toarray())
        if not isinstance(self, (TestBSR, TestCSC, TestCSR)):
            pytest.skip('Bool comparisons only implemented for BSR, CSC, and CSR.')
        for dtype in self.checked_dtypes:
            check(dtype)

    def test_ne(self):
        sup = suppress_warnings()
        sup.filter(SparseEfficiencyWarning)

        @sup
        @sup_complex
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            dat2 = dat.copy()
            dat2[:, 0] = 0
            datsp2 = self.spcreator(dat2)
            datbsr = bsr_matrix(dat)
            datcsc = csc_matrix(dat)
            datcsr = csr_matrix(dat)
            datlil = lil_matrix(dat)
            assert_array_equal_dtype(dat != dat2, (datsp != datsp2).toarray())
            assert_array_equal_dtype(dat != dat2, (datbsr != datsp2).toarray())
            assert_array_equal_dtype(dat != dat2, (datcsc != datsp2).toarray())
            assert_array_equal_dtype(dat != dat2, (datcsr != datsp2).toarray())
            assert_array_equal_dtype(dat != dat2, (datlil != datsp2).toarray())
            assert_array_equal_dtype(dat != datsp2, datsp2 != dat)
            assert_array_equal_dtype(dat != 0, (datsp != 0).toarray())
            assert_array_equal_dtype(dat != 1, (datsp != 1).toarray())
            assert_array_equal_dtype(0 != dat, (0 != datsp).toarray())
            assert_array_equal_dtype(1 != dat, (1 != datsp).toarray())
            assert_array_equal_dtype(dat != np.nan, (datsp != np.nan).toarray())
        if not isinstance(self, (TestBSR, TestCSC, TestCSR)):
            pytest.skip('Bool comparisons only implemented for BSR, CSC, and CSR.')
        for dtype in self.checked_dtypes:
            check(dtype)

    def test_lt(self):
        sup = suppress_warnings()
        sup.filter(SparseEfficiencyWarning)

        @sup
        @sup_complex
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            dat2 = dat.copy()
            dat2[:, 0] = 0
            datsp2 = self.spcreator(dat2)
            datcomplex = dat.astype(complex)
            datcomplex[:, 0] = 1 + 1j
            datspcomplex = self.spcreator(datcomplex)
            datbsr = bsr_matrix(dat)
            datcsc = csc_matrix(dat)
            datcsr = csr_matrix(dat)
            datlil = lil_matrix(dat)
            assert_array_equal_dtype(dat < dat2, (datsp < datsp2).toarray())
            assert_array_equal_dtype(datcomplex < dat2, (datspcomplex < datsp2).toarray())
            assert_array_equal_dtype(dat < dat2, (datbsr < datsp2).toarray())
            assert_array_equal_dtype(dat < dat2, (datcsc < datsp2).toarray())
            assert_array_equal_dtype(dat < dat2, (datcsr < datsp2).toarray())
            assert_array_equal_dtype(dat < dat2, (datlil < datsp2).toarray())
            assert_array_equal_dtype(dat2 < dat, (datsp2 < datbsr).toarray())
            assert_array_equal_dtype(dat2 < dat, (datsp2 < datcsc).toarray())
            assert_array_equal_dtype(dat2 < dat, (datsp2 < datcsr).toarray())
            assert_array_equal_dtype(dat2 < dat, (datsp2 < datlil).toarray())
            assert_array_equal_dtype(dat < dat2, datsp < dat2)
            assert_array_equal_dtype(datcomplex < dat2, datspcomplex < dat2)
            for val in [2, 1, 0, -1, -2]:
                val = np.int64(val)
                assert_array_equal_dtype((datsp < val).toarray(), dat < val)
                assert_array_equal_dtype((val < datsp).toarray(), val < dat)
            with np.errstate(invalid='ignore'):
                assert_array_equal_dtype((datsp < np.nan).toarray(), dat < np.nan)
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            dat2 = dat.copy()
            dat2[:, 0] = 0
            datsp2 = self.spcreator(dat2)
            assert_array_equal_dtype(dat < datsp2, datsp < dat2)
        if not isinstance(self, (TestBSR, TestCSC, TestCSR)):
            pytest.skip('Bool comparisons only implemented for BSR, CSC, and CSR.')
        for dtype in self.checked_dtypes:
            check(dtype)

    def test_gt(self):
        sup = suppress_warnings()
        sup.filter(SparseEfficiencyWarning)

        @sup
        @sup_complex
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            dat2 = dat.copy()
            dat2[:, 0] = 0
            datsp2 = self.spcreator(dat2)
            datcomplex = dat.astype(complex)
            datcomplex[:, 0] = 1 + 1j
            datspcomplex = self.spcreator(datcomplex)
            datbsr = bsr_matrix(dat)
            datcsc = csc_matrix(dat)
            datcsr = csr_matrix(dat)
            datlil = lil_matrix(dat)
            assert_array_equal_dtype(dat > dat2, (datsp > datsp2).toarray())
            assert_array_equal_dtype(datcomplex > dat2, (datspcomplex > datsp2).toarray())
            assert_array_equal_dtype(dat > dat2, (datbsr > datsp2).toarray())
            assert_array_equal_dtype(dat > dat2, (datcsc > datsp2).toarray())
            assert_array_equal_dtype(dat > dat2, (datcsr > datsp2).toarray())
            assert_array_equal_dtype(dat > dat2, (datlil > datsp2).toarray())
            assert_array_equal_dtype(dat2 > dat, (datsp2 > datbsr).toarray())
            assert_array_equal_dtype(dat2 > dat, (datsp2 > datcsc).toarray())
            assert_array_equal_dtype(dat2 > dat, (datsp2 > datcsr).toarray())
            assert_array_equal_dtype(dat2 > dat, (datsp2 > datlil).toarray())
            assert_array_equal_dtype(dat > dat2, datsp > dat2)
            assert_array_equal_dtype(datcomplex > dat2, datspcomplex > dat2)
            for val in [2, 1, 0, -1, -2]:
                val = np.int64(val)
                assert_array_equal_dtype((datsp > val).toarray(), dat > val)
                assert_array_equal_dtype((val > datsp).toarray(), val > dat)
            with np.errstate(invalid='ignore'):
                assert_array_equal_dtype((datsp > np.nan).toarray(), dat > np.nan)
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            dat2 = dat.copy()
            dat2[:, 0] = 0
            datsp2 = self.spcreator(dat2)
            assert_array_equal_dtype(dat > datsp2, datsp > dat2)
        if not isinstance(self, (TestBSR, TestCSC, TestCSR)):
            pytest.skip('Bool comparisons only implemented for BSR, CSC, and CSR.')
        for dtype in self.checked_dtypes:
            check(dtype)

    def test_le(self):
        sup = suppress_warnings()
        sup.filter(SparseEfficiencyWarning)

        @sup
        @sup_complex
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            dat2 = dat.copy()
            dat2[:, 0] = 0
            datsp2 = self.spcreator(dat2)
            datcomplex = dat.astype(complex)
            datcomplex[:, 0] = 1 + 1j
            datspcomplex = self.spcreator(datcomplex)
            datbsr = bsr_matrix(dat)
            datcsc = csc_matrix(dat)
            datcsr = csr_matrix(dat)
            datlil = lil_matrix(dat)
            assert_array_equal_dtype(dat <= dat2, (datsp <= datsp2).toarray())
            assert_array_equal_dtype(datcomplex <= dat2, (datspcomplex <= datsp2).toarray())
            assert_array_equal_dtype((datbsr <= datsp2).toarray(), dat <= dat2)
            assert_array_equal_dtype((datcsc <= datsp2).toarray(), dat <= dat2)
            assert_array_equal_dtype((datcsr <= datsp2).toarray(), dat <= dat2)
            assert_array_equal_dtype((datlil <= datsp2).toarray(), dat <= dat2)
            assert_array_equal_dtype((datsp2 <= datbsr).toarray(), dat2 <= dat)
            assert_array_equal_dtype((datsp2 <= datcsc).toarray(), dat2 <= dat)
            assert_array_equal_dtype((datsp2 <= datcsr).toarray(), dat2 <= dat)
            assert_array_equal_dtype((datsp2 <= datlil).toarray(), dat2 <= dat)
            assert_array_equal_dtype(datsp <= dat2, dat <= dat2)
            assert_array_equal_dtype(datspcomplex <= dat2, datcomplex <= dat2)
            for val in [2, 1, -1, -2]:
                val = np.int64(val)
                assert_array_equal_dtype((datsp <= val).toarray(), dat <= val)
                assert_array_equal_dtype((val <= datsp).toarray(), val <= dat)
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            dat2 = dat.copy()
            dat2[:, 0] = 0
            datsp2 = self.spcreator(dat2)
            assert_array_equal_dtype(dat <= datsp2, datsp <= dat2)
        if not isinstance(self, (TestBSR, TestCSC, TestCSR)):
            pytest.skip('Bool comparisons only implemented for BSR, CSC, and CSR.')
        for dtype in self.checked_dtypes:
            check(dtype)

    def test_ge(self):
        sup = suppress_warnings()
        sup.filter(SparseEfficiencyWarning)

        @sup
        @sup_complex
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            dat2 = dat.copy()
            dat2[:, 0] = 0
            datsp2 = self.spcreator(dat2)
            datcomplex = dat.astype(complex)
            datcomplex[:, 0] = 1 + 1j
            datspcomplex = self.spcreator(datcomplex)
            datbsr = bsr_matrix(dat)
            datcsc = csc_matrix(dat)
            datcsr = csr_matrix(dat)
            datlil = lil_matrix(dat)
            assert_array_equal_dtype(dat >= dat2, (datsp >= datsp2).toarray())
            assert_array_equal_dtype(datcomplex >= dat2, (datspcomplex >= datsp2).toarray())
            assert_array_equal_dtype((datbsr >= datsp2).toarray(), dat >= dat2)
            assert_array_equal_dtype((datcsc >= datsp2).toarray(), dat >= dat2)
            assert_array_equal_dtype((datcsr >= datsp2).toarray(), dat >= dat2)
            assert_array_equal_dtype((datlil >= datsp2).toarray(), dat >= dat2)
            assert_array_equal_dtype((datsp2 >= datbsr).toarray(), dat2 >= dat)
            assert_array_equal_dtype((datsp2 >= datcsc).toarray(), dat2 >= dat)
            assert_array_equal_dtype((datsp2 >= datcsr).toarray(), dat2 >= dat)
            assert_array_equal_dtype((datsp2 >= datlil).toarray(), dat2 >= dat)
            assert_array_equal_dtype(datsp >= dat2, dat >= dat2)
            assert_array_equal_dtype(datspcomplex >= dat2, datcomplex >= dat2)
            for val in [2, 1, -1, -2]:
                val = np.int64(val)
                assert_array_equal_dtype((datsp >= val).toarray(), dat >= val)
                assert_array_equal_dtype((val >= datsp).toarray(), val >= dat)
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            dat2 = dat.copy()
            dat2[:, 0] = 0
            datsp2 = self.spcreator(dat2)
            assert_array_equal_dtype(dat >= datsp2, datsp >= dat2)
        if not isinstance(self, (TestBSR, TestCSC, TestCSR)):
            pytest.skip('Bool comparisons only implemented for BSR, CSC, and CSR.')
        for dtype in self.checked_dtypes:
            check(dtype)

    def test_empty(self):
        assert_equal(self.spcreator((3, 3)).toarray(), zeros((3, 3)))
        assert_equal(self.spcreator((3, 3)).nnz, 0)
        assert_equal(self.spcreator((3, 3)).count_nonzero(), 0)

    def test_count_nonzero(self):
        expected = np.count_nonzero(self.datsp.toarray())
        assert_equal(self.datsp.count_nonzero(), expected)
        assert_equal(self.datsp.T.count_nonzero(), expected)

    def test_invalid_shapes(self):
        assert_raises(ValueError, self.spcreator, (-1, 3))
        assert_raises(ValueError, self.spcreator, (3, -1))
        assert_raises(ValueError, self.spcreator, (-1, -1))

    def test_repr(self):
        repr(self.datsp)

    def test_str(self):
        str(self.datsp)

    def test_empty_arithmetic(self):
        shape = (5, 5)
        for mytype in [np.dtype('int32'), np.dtype('float32'), np.dtype('float64'), np.dtype('complex64'), np.dtype('complex128')]:
            a = self.spcreator(shape, dtype=mytype)
            b = a + a
            c = 2 * a
            d = a @ a.tocsc()
            e = a @ a.tocsr()
            f = a @ a.tocoo()
            for m in [a, b, c, d, e, f]:
                assert_equal(m.toarray(), a.toarray() @ a.toarray())
                assert_equal(m.dtype, mytype)
                assert_equal(m.toarray().dtype, mytype)

    def test_abs(self):
        A = array([[-1, 0, 17], [0, -5, 0], [1, -4, 0], [0, 0, 0]], 'd')
        assert_equal(abs(A), abs(self.spcreator(A)).toarray())

    def test_round(self):
        decimal = 1
        A = array([[-1.35, 0.56], [17.25, -5.98]], 'd')
        assert_equal(np.around(A, decimals=decimal), round(self.spcreator(A), ndigits=decimal).toarray())

    def test_elementwise_power(self):
        A = array([[-4, -3, -2], [-1, 0, 1], [2, 3, 4]], 'd')
        assert_equal(np.power(A, 2), self.spcreator(A).power(2).toarray())
        assert_raises(NotImplementedError, self.spcreator(A).power, A)

    def test_neg(self):
        A = array([[-1, 0, 17], [0, -5, 0], [1, -4, 0], [0, 0, 0]], 'd')
        assert_equal(-A, (-self.spcreator(A)).toarray())
        A = array([[True, False, False], [False, False, True]])
        assert_raises(NotImplementedError, self.spcreator(A).__neg__)

    def test_real(self):
        D = array([[1 + 3j, 2 - 4j]])
        A = self.spcreator(D)
        assert_equal(A.real.toarray(), D.real)

    def test_imag(self):
        D = array([[1 + 3j, 2 - 4j]])
        A = self.spcreator(D)
        assert_equal(A.imag.toarray(), D.imag)

    def test_diagonal(self):
        mats = []
        mats.append([[1, 0, 2]])
        mats.append([[1], [0], [2]])
        mats.append([[0, 1], [0, 2], [0, 3]])
        mats.append([[0, 0, 1], [0, 0, 2], [0, 3, 0]])
        mats.append([[1, 0], [0, 0]])
        mats.append(kron(mats[0], [[1, 2]]))
        mats.append(kron(mats[0], [[1], [2]]))
        mats.append(kron(mats[1], [[1, 2], [3, 4]]))
        mats.append(kron(mats[2], [[1, 2], [3, 4]]))
        mats.append(kron(mats[3], [[1, 2], [3, 4]]))
        mats.append(kron(mats[3], [[1, 2, 3, 4]]))
        for m in mats:
            rows, cols = array(m).shape
            sparse_mat = self.spcreator(m)
            for k in range(-rows - 1, cols + 2):
                assert_equal(sparse_mat.diagonal(k=k), diag(m, k=k))
            assert_equal(sparse_mat.diagonal(k=10), diag(m, k=10))
            assert_equal(sparse_mat.diagonal(k=-99), diag(m, k=-99))
        assert_equal(self.spcreator((40, 16130)).diagonal(), np.zeros(40))
        assert_equal(self.spcreator((0, 0)).diagonal(), np.empty(0))
        assert_equal(self.spcreator((15, 0)).diagonal(), np.empty(0))
        assert_equal(self.spcreator((0, 5)).diagonal(10), np.empty(0))

    def test_trace(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = self.spcreator(A)
        for k in range(-2, 3):
            assert_equal(A.trace(offset=k), B.trace(offset=k))
        A = np.array([[1, 2, 3], [4, 5, 6]])
        B = self.spcreator(A)
        for k in range(-1, 3):
            assert_equal(A.trace(offset=k), B.trace(offset=k))

    def test_reshape(self):
        x = self.spcreator([[1, 0, 7], [0, 0, 0], [0, 3, 0], [0, 0, 5]])
        for order in ['C', 'F']:
            for s in [(12, 1), (1, 12)]:
                assert_array_equal(x.reshape(s, order=order).toarray(), x.toarray().reshape(s, order=order))
        x = self.spcreator([[0, 10, 0, 0], [0, 0, 0, 0], [0, 20, 30, 40]])
        y = x.reshape((2, 6))
        desired = [[0, 10, 0, 0, 0, 0], [0, 0, 0, 20, 30, 40]]
        assert_array_equal(y.toarray(), desired)
        y = x.reshape((2, -1))
        assert_array_equal(y.toarray(), desired)
        y = x.reshape((-1, 6))
        assert_array_equal(y.toarray(), desired)
        assert_raises(ValueError, x.reshape, (-1, -1))
        y = x.reshape(2, 6)
        assert_array_equal(y.toarray(), desired)
        assert_raises(TypeError, x.reshape, 2, 6, not_an_arg=1)
        y = x.reshape((3, 4))
        assert_(y is x)
        y = x.reshape((3, 4), copy=True)
        assert_(y is not x)
        assert_array_equal(x.shape, (3, 4))
        x.shape = (2, 6)
        assert_array_equal(x.toarray(), desired)
        assert_raises(ValueError, x.reshape, (x.size,))
        assert_raises(ValueError, x.reshape, (1, x.size, 1))

    @pytest.mark.slow
    def test_setdiag_comprehensive(self):

        def dense_setdiag(a, v, k):
            v = np.asarray(v)
            if k >= 0:
                n = min(a.shape[0], a.shape[1] - k)
                if v.ndim != 0:
                    n = min(n, len(v))
                    v = v[:n]
                i = np.arange(0, n)
                j = np.arange(k, k + n)
                a[i, j] = v
            elif k < 0:
                dense_setdiag(a.T, v, -k)

        def check_setdiag(a, b, k):
            for r in [-1, len(np.diag(a, k)), 2, 30]:
                if r < 0:
                    v = np.random.choice(range(1, 20))
                else:
                    v = np.random.randint(1, 20, size=r)
                dense_setdiag(a, v, k)
                with suppress_warnings() as sup:
                    message = 'Changing the sparsity structure of a cs[cr]_matrix is expensive'
                    sup.filter(SparseEfficiencyWarning, message)
                    b.setdiag(v, k)
                d = np.diag(a, k)
                if np.asarray(v).ndim == 0:
                    assert_array_equal(d, v, err_msg='%s %d' % (msg, r))
                else:
                    n = min(len(d), len(v))
                    assert_array_equal(d[:n], v[:n], err_msg='%s %d' % (msg, r))
                assert_array_equal(b.toarray(), a, err_msg='%s %d' % (msg, r))
        np.random.seed(1234)
        shapes = [(0, 5), (5, 0), (1, 5), (5, 1), (5, 5)]
        for dtype in [np.int8, np.float64]:
            for m, n in shapes:
                ks = np.arange(-m + 1, n - 1)
                for k in ks:
                    msg = repr((dtype, m, n, k))
                    a = np.zeros((m, n), dtype=dtype)
                    b = self.spcreator((m, n), dtype=dtype)
                    check_setdiag(a, b, k)
                    for k2 in np.random.choice(ks, size=min(len(ks), 5)):
                        check_setdiag(a, b, k2)

    def test_setdiag(self):
        m = self.spcreator(np.eye(3))
        m2 = self.spcreator((4, 4))
        values = [3, 2, 1]
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
            assert_raises(ValueError, m.setdiag, values, k=4)
            m.setdiag(values)
            assert_array_equal(m.diagonal(), values)
            m.setdiag(values, k=1)
            assert_array_equal(m.toarray(), np.array([[3, 3, 0], [0, 2, 2], [0, 0, 1]]))
            m.setdiag(values, k=-2)
            assert_array_equal(m.toarray(), np.array([[3, 3, 0], [0, 2, 2], [3, 0, 1]]))
            m.setdiag((9,), k=2)
            assert_array_equal(m.toarray()[0, 2], 9)
            m.setdiag((9,), k=-2)
            assert_array_equal(m.toarray()[2, 0], 9)
            m2.setdiag([1], k=2)
            assert_array_equal(m2.toarray()[0], [0, 0, 1, 0])
            m2.setdiag([1, 1], k=2)
            assert_array_equal(m2.toarray()[:2], [[0, 0, 1, 0], [0, 0, 0, 1]])

    def test_nonzero(self):
        A = array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
        Asp = self.spcreator(A)
        A_nz = {tuple(ij) for ij in transpose(A.nonzero())}
        Asp_nz = {tuple(ij) for ij in transpose(Asp.nonzero())}
        assert_equal(A_nz, Asp_nz)

    def test_numpy_nonzero(self):
        A = array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
        Asp = self.spcreator(A)
        A_nz = {tuple(ij) for ij in transpose(np.nonzero(A))}
        Asp_nz = {tuple(ij) for ij in transpose(np.nonzero(Asp))}
        assert_equal(A_nz, Asp_nz)

    def test_getrow(self):
        assert_array_equal(self.datsp.getrow(1).toarray(), self.dat[[1], :])
        assert_array_equal(self.datsp.getrow(-1).toarray(), self.dat[[-1], :])

    def test_getcol(self):
        assert_array_equal(self.datsp.getcol(1).toarray(), self.dat[:, [1]])
        assert_array_equal(self.datsp.getcol(-1).toarray(), self.dat[:, [-1]])

    def test_sum(self):
        np.random.seed(1234)
        dat_1 = matrix([[0, 1, 2], [3, -4, 5], [-6, 7, 9]])
        dat_2 = np.random.rand(5, 5)
        dat_3 = np.array([[]])
        dat_4 = np.zeros((40, 40))
        dat_5 = sparse.rand(5, 5, density=0.01).toarray()
        matrices = [dat_1, dat_2, dat_3, dat_4, dat_5]

        def check(dtype, j):
            dat = matrix(matrices[j], dtype=dtype)
            datsp = self.spcreator(dat, dtype=dtype)
            with np.errstate(over='ignore'):
                assert_array_almost_equal(dat.sum(), datsp.sum())
                assert_equal(dat.sum().dtype, datsp.sum().dtype)
                assert_(np.isscalar(datsp.sum(axis=None)))
                assert_array_almost_equal(dat.sum(axis=None), datsp.sum(axis=None))
                assert_equal(dat.sum(axis=None).dtype, datsp.sum(axis=None).dtype)
                assert_array_almost_equal(dat.sum(axis=0), datsp.sum(axis=0))
                assert_equal(dat.sum(axis=0).dtype, datsp.sum(axis=0).dtype)
                assert_array_almost_equal(dat.sum(axis=1), datsp.sum(axis=1))
                assert_equal(dat.sum(axis=1).dtype, datsp.sum(axis=1).dtype)
                assert_array_almost_equal(dat.sum(axis=-2), datsp.sum(axis=-2))
                assert_equal(dat.sum(axis=-2).dtype, datsp.sum(axis=-2).dtype)
                assert_array_almost_equal(dat.sum(axis=-1), datsp.sum(axis=-1))
                assert_equal(dat.sum(axis=-1).dtype, datsp.sum(axis=-1).dtype)
        for dtype in self.checked_dtypes:
            for j in range(len(matrices)):
                check(dtype, j)

    def test_sum_invalid_params(self):
        out = np.zeros((1, 3))
        dat = array([[0, 1, 2], [3, -4, 5], [-6, 7, 9]])
        datsp = self.spcreator(dat)
        assert_raises(ValueError, datsp.sum, axis=3)
        assert_raises(TypeError, datsp.sum, axis=(0, 1))
        assert_raises(TypeError, datsp.sum, axis=1.5)
        assert_raises(ValueError, datsp.sum, axis=1, out=out)

    def test_sum_dtype(self):
        dat = array([[0, 1, 2], [3, -4, 5], [-6, 7, 9]])
        datsp = self.spcreator(dat)

        def check(dtype):
            dat_mean = dat.mean(dtype=dtype)
            datsp_mean = datsp.mean(dtype=dtype)
            assert_array_almost_equal(dat_mean, datsp_mean)
            assert_equal(dat_mean.dtype, datsp_mean.dtype)
        for dtype in self.checked_dtypes:
            check(dtype)

    def test_sum_out(self):
        dat = array([[0, 1, 2], [3, -4, 5], [-6, 7, 9]])
        datsp = self.spcreator(dat)
        dat_out = array([[0]])
        datsp_out = matrix([[0]])
        dat.sum(out=dat_out, keepdims=True)
        datsp.sum(out=datsp_out)
        assert_array_almost_equal(dat_out, datsp_out)
        dat_out = np.zeros((3, 1))
        datsp_out = asmatrix(np.zeros((3, 1)))
        dat.sum(axis=1, out=dat_out, keepdims=True)
        datsp.sum(axis=1, out=datsp_out)
        assert_array_almost_equal(dat_out, datsp_out)

    def test_numpy_sum(self):
        dat = array([[0, 1, 2], [3, -4, 5], [-6, 7, 9]])
        datsp = self.spcreator(dat)
        dat_mean = np.sum(dat)
        datsp_mean = np.sum(datsp)
        assert_array_almost_equal(dat_mean, datsp_mean)
        assert_equal(dat_mean.dtype, datsp_mean.dtype)

    def test_mean(self):

        def check(dtype):
            dat = array([[0, 1, 2], [3, 4, 5], [6, 7, 9]], dtype=dtype)
            datsp = self.spcreator(dat, dtype=dtype)
            assert_array_almost_equal(dat.mean(), datsp.mean())
            assert_equal(dat.mean().dtype, datsp.mean().dtype)
            assert_(np.isscalar(datsp.mean(axis=None)))
            assert_array_almost_equal(dat.mean(axis=None, keepdims=True), datsp.mean(axis=None))
            assert_equal(dat.mean(axis=None).dtype, datsp.mean(axis=None).dtype)
            assert_array_almost_equal(dat.mean(axis=0, keepdims=True), datsp.mean(axis=0))
            assert_equal(dat.mean(axis=0).dtype, datsp.mean(axis=0).dtype)
            assert_array_almost_equal(dat.mean(axis=1, keepdims=True), datsp.mean(axis=1))
            assert_equal(dat.mean(axis=1).dtype, datsp.mean(axis=1).dtype)
            assert_array_almost_equal(dat.mean(axis=-2, keepdims=True), datsp.mean(axis=-2))
            assert_equal(dat.mean(axis=-2).dtype, datsp.mean(axis=-2).dtype)
            assert_array_almost_equal(dat.mean(axis=-1, keepdims=True), datsp.mean(axis=-1))
            assert_equal(dat.mean(axis=-1).dtype, datsp.mean(axis=-1).dtype)
        for dtype in self.checked_dtypes:
            check(dtype)

    def test_mean_invalid_params(self):
        out = asmatrix(np.zeros((1, 3)))
        dat = array([[0, 1, 2], [3, -4, 5], [-6, 7, 9]])
        datsp = self.spcreator(dat)
        assert_raises(ValueError, datsp.mean, axis=3)
        assert_raises(TypeError, datsp.mean, axis=(0, 1))
        assert_raises(TypeError, datsp.mean, axis=1.5)
        assert_raises(ValueError, datsp.mean, axis=1, out=out)

    def test_mean_dtype(self):
        dat = array([[0, 1, 2], [3, -4, 5], [-6, 7, 9]])
        datsp = self.spcreator(dat)

        def check(dtype):
            dat_mean = dat.mean(dtype=dtype)
            datsp_mean = datsp.mean(dtype=dtype)
            assert_array_almost_equal(dat_mean, datsp_mean)
            assert_equal(dat_mean.dtype, datsp_mean.dtype)
        for dtype in self.checked_dtypes:
            check(dtype)

    def test_mean_out(self):
        dat = array([[0, 1, 2], [3, -4, 5], [-6, 7, 9]])
        datsp = self.spcreator(dat)
        dat_out = array([[0]])
        datsp_out = matrix([[0]])
        dat.mean(out=dat_out, keepdims=True)
        datsp.mean(out=datsp_out)
        assert_array_almost_equal(dat_out, datsp_out)
        dat_out = np.zeros((3, 1))
        datsp_out = matrix(np.zeros((3, 1)))
        dat.mean(axis=1, out=dat_out, keepdims=True)
        datsp.mean(axis=1, out=datsp_out)
        assert_array_almost_equal(dat_out, datsp_out)

    def test_numpy_mean(self):
        dat = array([[0, 1, 2], [3, -4, 5], [-6, 7, 9]])
        datsp = self.spcreator(dat)
        dat_mean = np.mean(dat)
        datsp_mean = np.mean(datsp)
        assert_array_almost_equal(dat_mean, datsp_mean)
        assert_equal(dat_mean.dtype, datsp_mean.dtype)

    def test_expm(self):
        M = array([[1, 0, 2], [0, 0, 3], [-4, 5, 6]], float)
        sM = self.spcreator(M, shape=(3, 3), dtype=float)
        Mexp = scipy.linalg.expm(M)
        N = array([[3.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        sN = self.spcreator(N, shape=(3, 3), dtype=float)
        Nexp = scipy.linalg.expm(N)
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'splu converted its input to CSC format')
            sup.filter(SparseEfficiencyWarning, 'spsolve is more efficient when sparse b is in the CSC matrix format')
            sup.filter(SparseEfficiencyWarning, 'spsolve requires A be CSC or CSR matrix format')
            sMexp = expm(sM).toarray()
            sNexp = expm(sN).toarray()
        assert_array_almost_equal(sMexp - Mexp, zeros((3, 3)))
        assert_array_almost_equal(sNexp - Nexp, zeros((3, 3)))

    def test_inv(self):

        def check(dtype):
            M = array([[1, 0, 2], [0, 0, 3], [-4, 5, 6]], dtype)
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, 'spsolve requires A be CSC or CSR matrix format')
                sup.filter(SparseEfficiencyWarning, 'spsolve is more efficient when sparse b is in the CSC matrix format')
                sup.filter(SparseEfficiencyWarning, 'splu converted its input to CSC format')
                sM = self.spcreator(M, shape=(3, 3), dtype=dtype)
                sMinv = inv(sM)
            assert_array_almost_equal(sMinv.dot(sM).toarray(), np.eye(3))
            assert_raises(TypeError, inv, M)
        for dtype in [float]:
            check(dtype)

    @sup_complex
    def test_from_array(self):
        A = array([[1, 0, 0], [2, 3, 4], [0, 5, 0], [0, 0, 0]])
        assert_array_equal(self.spcreator(A).toarray(), A)
        A = array([[1.0 + 3j, 0, 0], [0, 2.0 + 5, 0], [0, 0, 0]])
        assert_array_equal(self.spcreator(A).toarray(), A)
        assert_array_equal(self.spcreator(A, dtype='int16').toarray(), A.astype('int16'))

    @sup_complex
    def test_from_matrix(self):
        A = matrix([[1, 0, 0], [2, 3, 4], [0, 5, 0], [0, 0, 0]])
        assert_array_equal(self.spcreator(A).todense(), A)
        A = matrix([[1.0 + 3j, 0, 0], [0, 2.0 + 5, 0], [0, 0, 0]])
        assert_array_equal(self.spcreator(A).todense(), A)
        assert_array_equal(self.spcreator(A, dtype='int16').todense(), A.astype('int16'))

    @sup_complex
    def test_from_list(self):
        A = [[1, 0, 0], [2, 3, 4], [0, 5, 0], [0, 0, 0]]
        assert_array_equal(self.spcreator(A).toarray(), A)
        A = [[1.0 + 3j, 0, 0], [0, 2.0 + 5, 0], [0, 0, 0]]
        assert_array_equal(self.spcreator(A).toarray(), array(A))
        assert_array_equal(self.spcreator(A, dtype='int16').toarray(), array(A).astype('int16'))

    @sup_complex
    def test_from_sparse(self):
        D = array([[1, 0, 0], [2, 3, 4], [0, 5, 0], [0, 0, 0]])
        S = csr_matrix(D)
        assert_array_equal(self.spcreator(S).toarray(), D)
        S = self.spcreator(D)
        assert_array_equal(self.spcreator(S).toarray(), D)
        D = array([[1.0 + 3j, 0, 0], [0, 2.0 + 5, 0], [0, 0, 0]])
        S = csr_matrix(D)
        assert_array_equal(self.spcreator(S).toarray(), D)
        assert_array_equal(self.spcreator(S, dtype='int16').toarray(), D.astype('int16'))
        S = self.spcreator(D)
        assert_array_equal(self.spcreator(S).toarray(), D)
        assert_array_equal(self.spcreator(S, dtype='int16').toarray(), D.astype('int16'))

    def test_todense(self):
        chk = self.datsp.todense()
        assert isinstance(chk, np.matrix)
        assert_array_equal(chk, self.dat)
        assert_(chk.flags.c_contiguous != chk.flags.f_contiguous)
        chk = self.datsp.todense(order='C')
        assert_array_equal(chk, self.dat)
        assert_(chk.flags.c_contiguous)
        assert_(not chk.flags.f_contiguous)
        chk = self.datsp.todense(order='F')
        assert_array_equal(chk, self.dat)
        assert_(not chk.flags.c_contiguous)
        assert_(chk.flags.f_contiguous)
        out = np.zeros(self.datsp.shape, dtype=self.datsp.dtype)
        chk = self.datsp.todense(out=out)
        assert_array_equal(self.dat, out)
        assert_array_equal(self.dat, chk)
        assert_(chk.base is out)
        out = asmatrix(np.zeros(self.datsp.shape, dtype=self.datsp.dtype))
        chk = self.datsp.todense(out=out)
        assert_array_equal(self.dat, out)
        assert_array_equal(self.dat, chk)
        assert_(chk is out)
        a = array([[1.0, 2.0, 3.0]])
        dense_dot_dense = a @ self.dat
        check = a @ self.datsp.todense()
        assert_array_equal(dense_dot_dense, check)
        b = array([[1.0, 2.0, 3.0, 4.0]]).T
        dense_dot_dense = self.dat @ b
        check2 = self.datsp.todense() @ b
        assert_array_equal(dense_dot_dense, check2)
        spbool = self.spcreator(self.dat, dtype=bool)
        matbool = self.dat.astype(bool)
        assert_array_equal(spbool.todense(), matbool)

    def test_toarray(self):
        dat = asarray(self.dat)
        chk = self.datsp.toarray()
        assert_array_equal(chk, dat)
        assert_(chk.flags.c_contiguous != chk.flags.f_contiguous)
        chk = self.datsp.toarray(order='C')
        assert_array_equal(chk, dat)
        assert_(chk.flags.c_contiguous)
        assert_(not chk.flags.f_contiguous)
        chk = self.datsp.toarray(order='F')
        assert_array_equal(chk, dat)
        assert_(not chk.flags.c_contiguous)
        assert_(chk.flags.f_contiguous)
        out = np.zeros(self.datsp.shape, dtype=self.datsp.dtype)
        self.datsp.toarray(out=out)
        assert_array_equal(chk, dat)
        out[...] = 1.0
        self.datsp.toarray(out=out)
        assert_array_equal(chk, dat)
        a = array([1.0, 2.0, 3.0])
        dense_dot_dense = dot(a, dat)
        check = dot(a, self.datsp.toarray())
        assert_array_equal(dense_dot_dense, check)
        b = array([1.0, 2.0, 3.0, 4.0])
        dense_dot_dense = dot(dat, b)
        check2 = dot(self.datsp.toarray(), b)
        assert_array_equal(dense_dot_dense, check2)
        spbool = self.spcreator(self.dat, dtype=bool)
        arrbool = dat.astype(bool)
        assert_array_equal(spbool.toarray(), arrbool)

    @sup_complex
    def test_astype(self):
        D = array([[2.0 + 3j, 0, 0], [0, 4.0 + 5j, 0], [0, 0, 0]])
        S = self.spcreator(D)
        for x in supported_dtypes:
            D_casted = D.astype(x)
            for copy in (True, False):
                S_casted = S.astype(x, copy=copy)
                assert_equal(S_casted.dtype, D_casted.dtype)
                assert_equal(S_casted.toarray(), D_casted)
                assert_equal(S_casted.format, S.format)
            assert_(S_casted.astype(x, copy=False) is S_casted)
            S_copied = S_casted.astype(x, copy=True)
            assert_(S_copied is not S_casted)

            def check_equal_but_not_same_array_attribute(attribute):
                a = getattr(S_casted, attribute)
                b = getattr(S_copied, attribute)
                assert_array_equal(a, b)
                assert_(a is not b)
                i = (0,) * b.ndim
                b_i = b[i]
                b[i] = not b[i]
                assert_(a[i] != b[i])
                b[i] = b_i
            if S_casted.format in ('csr', 'csc', 'bsr'):
                for attribute in ('indices', 'indptr', 'data'):
                    check_equal_but_not_same_array_attribute(attribute)
            elif S_casted.format == 'coo':
                for attribute in ('row', 'col', 'data'):
                    check_equal_but_not_same_array_attribute(attribute)
            elif S_casted.format == 'dia':
                for attribute in ('offsets', 'data'):
                    check_equal_but_not_same_array_attribute(attribute)

    @sup_complex
    def test_astype_immutable(self):
        D = array([[2.0 + 3j, 0, 0], [0, 4.0 + 5j, 0], [0, 0, 0]])
        S = self.spcreator(D)
        if hasattr(S, 'data'):
            S.data.flags.writeable = False
        if hasattr(S, 'indptr'):
            S.indptr.flags.writeable = False
        if hasattr(S, 'indices'):
            S.indices.flags.writeable = False
        for x in supported_dtypes:
            D_casted = D.astype(x)
            S_casted = S.astype(x)
            assert_equal(S_casted.dtype, D_casted.dtype)

    def test_asfptype(self):
        A = self.spcreator(arange(6, dtype='int32').reshape(2, 3))
        assert_equal(A.dtype, np.dtype('int32'))
        assert_equal(A.asfptype().dtype, np.dtype('float64'))
        assert_equal(A.asfptype().format, A.format)
        assert_equal(A.astype('int16').asfptype().dtype, np.dtype('float32'))
        assert_equal(A.astype('complex128').asfptype().dtype, np.dtype('complex128'))
        B = A.asfptype()
        C = B.asfptype()
        assert_(B is C)

    def test_mul_scalar(self):

        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            assert_array_equal(dat * 2, (datsp * 2).toarray())
            assert_array_equal(dat * 17.3, (datsp * 17.3).toarray())
        for dtype in self.math_dtypes:
            check(dtype)

    def test_rmul_scalar(self):

        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            assert_array_equal(2 * dat, (2 * datsp).toarray())
            assert_array_equal(17.3 * dat, (17.3 * datsp).toarray())
        for dtype in self.math_dtypes:
            check(dtype)

    def test_rmul_scalar_type_error(self):
        datsp = self.datsp_dtypes[np.float64]
        with assert_raises(TypeError):
            None * datsp

    def test_add(self):

        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            a = dat.copy()
            a[0, 2] = 2.0
            b = datsp
            c = b + a
            assert_array_equal(c, b.toarray() + a)
            c = b + b.tocsr()
            assert_array_equal(c.toarray(), b.toarray() + b.toarray())
            c = b + a[0]
            assert_array_equal(c, b.toarray() + a[0])
        for dtype in self.math_dtypes:
            check(dtype)

    def test_radd(self):

        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            a = dat.copy()
            a[0, 2] = 2.0
            b = datsp
            c = a + b
            assert_array_equal(c, a + b.toarray())
        for dtype in self.math_dtypes:
            check(dtype)

    def test_sub(self):

        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            assert_array_equal((datsp - datsp).toarray(), np.zeros((3, 4)))
            assert_array_equal((datsp - 0).toarray(), dat)
            A = self.spcreator(np.array([[1, 0, 0, 4], [-1, 0, 0, 0], [0, 8, 0, -5]], 'd'))
            assert_array_equal((datsp - A).toarray(), dat - A.toarray())
            assert_array_equal((A - datsp).toarray(), A.toarray() - dat)
            assert_array_equal(datsp - dat[0], dat - dat[0])
        for dtype in self.math_dtypes:
            if dtype == np.dtype('bool'):
                continue
            check(dtype)

    def test_rsub(self):

        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            assert_array_equal(dat - datsp, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
            assert_array_equal(datsp - dat, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
            assert_array_equal((0 - datsp).toarray(), -dat)
            A = self.spcreator(matrix([[1, 0, 0, 4], [-1, 0, 0, 0], [0, 8, 0, -5]], 'd'))
            assert_array_equal(dat - A, dat - A.toarray())
            assert_array_equal(A - dat, A.toarray() - dat)
            assert_array_equal(A.toarray() - datsp, A.toarray() - dat)
            assert_array_equal(datsp - A.toarray(), dat - A.toarray())
            assert_array_equal(dat[0] - datsp, dat[0] - dat)
        for dtype in self.math_dtypes:
            if dtype == np.dtype('bool'):
                continue
            check(dtype)

    def test_add0(self):

        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            assert_array_equal((datsp + 0).toarray(), dat)
            sumS = sum([k * datsp for k in range(1, 3)])
            sumD = sum([k * dat for k in range(1, 3)])
            assert_almost_equal(sumS.toarray(), sumD)
        for dtype in self.math_dtypes:
            check(dtype)

    def test_elementwise_multiply(self):
        A = array([[4, 0, 9], [2, -3, 5]])
        B = array([[0, 7, 0], [0, -4, 0]])
        Asp = self.spcreator(A)
        Bsp = self.spcreator(B)
        assert_almost_equal(Asp.multiply(Bsp).toarray(), A * B)
        assert_almost_equal(Asp.multiply(B).toarray(), A * B)
        C = array([[1 - 2j, 0 + 5j, -1 + 0j], [4 - 3j, -3 + 6j, 5]])
        D = array([[5 + 2j, 7 - 3j, -2 + 1j], [0 - 1j, -4 + 2j, 9]])
        Csp = self.spcreator(C)
        Dsp = self.spcreator(D)
        assert_almost_equal(Csp.multiply(Dsp).toarray(), C * D)
        assert_almost_equal(Csp.multiply(D).toarray(), C * D)
        assert_almost_equal(Asp.multiply(Dsp).toarray(), A * D)
        assert_almost_equal(Asp.multiply(D).toarray(), A * D)

    def test_elementwise_multiply_broadcast(self):
        A = array([4])
        B = array([[-9]])
        C = array([1, -1, 0])
        D = array([[7, 9, -9]])
        E = array([[3], [2], [1]])
        F = array([[8, 6, 3], [-4, 3, 2], [6, 6, 6]])
        G = [1, 2, 3]
        H = np.ones((3, 4))
        J = H.T
        K = array([[0]])
        L = array([[[1, 2], [0, 1]]])
        Bsp = self.spcreator(B)
        Dsp = self.spcreator(D)
        Esp = self.spcreator(E)
        Fsp = self.spcreator(F)
        Hsp = self.spcreator(H)
        Hspp = self.spcreator(H[0, None])
        Jsp = self.spcreator(J)
        Jspp = self.spcreator(J[:, 0, None])
        Ksp = self.spcreator(K)
        matrices = [A, B, C, D, E, F, G, H, J, K, L]
        spmatrices = [Bsp, Dsp, Esp, Fsp, Hsp, Hspp, Jsp, Jspp, Ksp]
        for i in spmatrices:
            for j in spmatrices:
                try:
                    dense_mult = i.toarray() * j.toarray()
                except ValueError:
                    assert_raises(ValueError, i.multiply, j)
                    continue
                sp_mult = i.multiply(j)
                assert_almost_equal(sp_mult.toarray(), dense_mult)
        for i in spmatrices:
            for j in matrices:
                try:
                    dense_mult = i.toarray() * j
                except TypeError:
                    continue
                except ValueError:
                    assert_raises(ValueError, i.multiply, j)
                    continue
                sp_mult = i.multiply(j)
                if issparse(sp_mult):
                    assert_almost_equal(sp_mult.toarray(), dense_mult)
                else:
                    assert_almost_equal(sp_mult, dense_mult)

    def test_elementwise_divide(self):
        expected = [[1, np.nan, np.nan, 1], [1, np.nan, 1, np.nan], [np.nan, 1, np.nan, np.nan]]
        assert_array_equal(toarray(self.datsp / self.datsp), expected)
        denom = self.spcreator(matrix([[1, 0, 0, 4], [-1, 0, 0, 0], [0, 8, 0, -5]], 'd'))
        expected = [[1, np.nan, np.nan, 0.5], [-3, np.nan, inf, np.nan], [np.nan, 0.25, np.nan, 0]]
        assert_array_equal(toarray(self.datsp / denom), expected)
        A = array([[1 - 2j, 0 + 5j, -1 + 0j], [4 - 3j, -3 + 6j, 5]])
        B = array([[5 + 2j, 7 - 3j, -2 + 1j], [0 - 1j, -4 + 2j, 9]])
        Asp = self.spcreator(A)
        Bsp = self.spcreator(B)
        assert_almost_equal(toarray(Asp / Bsp), A / B)
        A = array([[1, 2, 3], [-3, 2, 1]])
        B = array([[0, 1, 2], [0, -2, 3]])
        Asp = self.spcreator(A)
        Bsp = self.spcreator(B)
        with np.errstate(divide='ignore'):
            assert_array_equal(toarray(Asp / Bsp), A / B)
        A = array([[0, 1], [1, 0]])
        B = array([[1, 0], [1, 0]])
        Asp = self.spcreator(A)
        Bsp = self.spcreator(B)
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_array_equal(np.array(toarray(Asp / Bsp)), A / B)

    def test_pow(self):
        A = array([[1, 0, 2, 0], [0, 3, 4, 0], [0, 5, 0, 0], [0, 6, 7, 8]])
        B = self.spcreator(A)
        for exponent in [0, 1, 2, 3]:
            ret_sp = B ** exponent
            ret_np = np.linalg.matrix_power(A, exponent)
            assert_array_equal(ret_sp.toarray(), ret_np)
            assert_equal(ret_sp.dtype, ret_np.dtype)
        for exponent in [-1, 2.2, 1 + 3j]:
            assert_raises(ValueError, B.__pow__, exponent)
        B = self.spcreator(A[:3, :])
        assert_raises(TypeError, B.__pow__, 1)

    def test_rmatvec(self):
        M = self.spcreator(matrix([[3, 0, 0], [0, 1, 0], [2, 0, 3.0], [2, 3, 0]]))
        assert_array_almost_equal([1, 2, 3, 4] @ M, dot([1, 2, 3, 4], M.toarray()))
        row = array([[1, 2, 3, 4]])
        assert_array_almost_equal(row @ M, row @ M.toarray())

    def test_small_multiplication(self):
        A = self.spcreator([[1], [2], [3]])
        assert_(issparse(A * array(1)))
        assert_equal((A * array(1)).toarray(), [[1], [2], [3]])
        assert_equal(A @ array([1]), array([1, 2, 3]))
        assert_equal(A @ array([[1]]), array([[1], [2], [3]]))
        assert_equal(A @ np.ones((1, 1)), array([[1], [2], [3]]))
        assert_equal(A @ np.ones((1, 0)), np.ones((3, 0)))

    def test_start_vs_at_sign_for_sparray_and_spmatrix(self):
        A = self.spcreator([[1], [2], [3]])
        if isinstance(A, sparray):
            assert_array_almost_equal(A * np.ones((3, 1)), A)
            assert_array_almost_equal(A * array([[1]]), A)
            assert_array_almost_equal(A * np.ones((3, 1)), A)
        else:
            assert_equal(A * array([1]), array([1, 2, 3]))
            assert_equal(A * array([[1]]), array([[1], [2], [3]]))
            assert_equal(A * np.ones((1, 0)), np.ones((3, 0)))

    def test_binop_custom_type(self):
        A = self.spcreator([[1], [2], [3]])
        B = BinopTester()
        assert_equal(A + B, 'matrix on the left')
        assert_equal(A - B, 'matrix on the left')
        assert_equal(A * B, 'matrix on the left')
        assert_equal(B + A, 'matrix on the right')
        assert_equal(B - A, 'matrix on the right')
        assert_equal(B * A, 'matrix on the right')
        assert_equal(A @ B, 'matrix on the left')
        assert_equal(B @ A, 'matrix on the right')

    def test_binop_custom_type_with_shape(self):
        A = self.spcreator([[1], [2], [3]])
        B = BinopTester_with_shape((3, 1))
        assert_equal(A + B, 'matrix on the left')
        assert_equal(A - B, 'matrix on the left')
        assert_equal(A * B, 'matrix on the left')
        assert_equal(B + A, 'matrix on the right')
        assert_equal(B - A, 'matrix on the right')
        assert_equal(B * A, 'matrix on the right')
        assert_equal(A @ B, 'matrix on the left')
        assert_equal(B @ A, 'matrix on the right')

    def test_mul_custom_type(self):

        class Custom:

            def __init__(self, scalar):
                self.scalar = scalar

            def __rmul__(self, other):
                return other * self.scalar
        scalar = 2
        A = self.spcreator([[1], [2], [3]])
        c = Custom(scalar)
        A_scalar = A * scalar
        A_c = A * c
        assert_array_equal_dtype(A_scalar.toarray(), A_c.toarray())
        assert_equal(A_scalar.format, A_c.format)

    def test_comparisons_custom_type(self):
        A = self.spcreator([[1], [2], [3]])
        B = ComparisonTester()
        assert_equal(A == B, 'eq')
        assert_equal(A != B, 'ne')
        assert_equal(A > B, 'lt')
        assert_equal(A >= B, 'le')
        assert_equal(A < B, 'gt')
        assert_equal(A <= B, 'ge')

    def test_dot_scalar(self):
        M = self.spcreator(array([[3, 0, 0], [0, 1, 0], [2, 0, 3.0], [2, 3, 0]]))
        scalar = 10
        actual = M.dot(scalar)
        expected = M * scalar
        assert_allclose(actual.toarray(), expected.toarray())

    def test_matmul(self):
        M = self.spcreator(array([[3, 0, 0], [0, 1, 0], [2, 0, 3.0], [2, 3, 0]]))
        B = self.spcreator(array([[0, 1], [1, 0], [0, 2]], 'd'))
        col = array([[1, 2, 3]]).T
        matmul = operator.matmul
        assert_array_almost_equal(matmul(M, col), M.toarray() @ col)
        assert_array_almost_equal(matmul(M, B).toarray(), (M @ B).toarray())
        assert_array_almost_equal(matmul(M.toarray(), B), (M @ B).toarray())
        assert_array_almost_equal(matmul(M, B.toarray()), (M @ B).toarray())
        if not isinstance(M, sparray):
            assert_array_almost_equal(matmul(M, B).toarray(), (M * B).toarray())
            assert_array_almost_equal(matmul(M.toarray(), B), (M * B).toarray())
            assert_array_almost_equal(matmul(M, B.toarray()), (M * B).toarray())
        assert_raises(ValueError, matmul, M, 1)
        assert_raises(ValueError, matmul, 1, M)

    def test_matvec(self):
        M = self.spcreator(matrix([[3, 0, 0], [0, 1, 0], [2, 0, 3.0], [2, 3, 0]]))
        col = array([[1, 2, 3]]).T
        assert_array_almost_equal(M @ col, M.toarray() @ col)
        assert_equal((M @ array([1, 2, 3])).shape, (4,))
        assert_equal((M @ array([[1], [2], [3]])).shape, (4, 1))
        assert_equal((M @ matrix([[1], [2], [3]])).shape, (4, 1))
        assert_(isinstance(M @ array([1, 2, 3]), ndarray))
        assert_(isinstance(M @ matrix([1, 2, 3]).T, np.matrix))
        bad_vecs = [array([1, 2]), array([1, 2, 3, 4]), array([[1], [2]]), matrix([1, 2, 3]), matrix([[1], [2]])]
        for x in bad_vecs:
            assert_raises(ValueError, M.__mul__, x)
        assert_array_almost_equal(M @ array([1, 2, 3]), dot(M.toarray(), [1, 2, 3]))
        assert_array_almost_equal(M @ [[1], [2], [3]], asmatrix(dot(M.toarray(), [1, 2, 3])).T)

    def test_matmat_sparse(self):
        a = matrix([[3, 0, 0], [0, 1, 0], [2, 0, 3.0], [2, 3, 0]])
        a2 = array([[3, 0, 0], [0, 1, 0], [2, 0, 3.0], [2, 3, 0]])
        b = matrix([[0, 1], [1, 0], [0, 2]], 'd')
        asp = self.spcreator(a)
        bsp = self.spcreator(b)
        assert_array_almost_equal((asp @ bsp).toarray(), a @ b)
        assert_array_almost_equal(asp @ b, a @ b)
        assert_array_almost_equal(a @ bsp, a @ b)
        assert_array_almost_equal(a2 @ bsp, a @ b)
        csp = bsp.tocsc()
        c = b
        want = a @ c
        assert_array_almost_equal((asp @ csp).toarray(), want)
        assert_array_almost_equal(asp @ c, want)
        assert_array_almost_equal(a @ csp, want)
        assert_array_almost_equal(a2 @ csp, want)
        csp = bsp.tocsr()
        assert_array_almost_equal((asp @ csp).toarray(), want)
        assert_array_almost_equal(asp @ c, want)
        assert_array_almost_equal(a @ csp, want)
        assert_array_almost_equal(a2 @ csp, want)
        csp = bsp.tocoo()
        assert_array_almost_equal((asp @ csp).toarray(), want)
        assert_array_almost_equal(asp @ c, want)
        assert_array_almost_equal(a @ csp, want)
        assert_array_almost_equal(a2 @ csp, want)
        L = 30
        frac = 0.3
        random.seed(0)
        A = zeros((L, 2))
        for i in range(L):
            for j in range(2):
                r = random.random()
                if r < frac:
                    A[i, j] = r / frac
        A = self.spcreator(A)
        B = A @ A.T
        assert_array_almost_equal(B.toarray(), A.toarray() @ A.T.toarray())
        assert_array_almost_equal(B.toarray(), A.toarray() @ A.toarray().T)
        A = self.spcreator([[1, 2], [3, 4]])
        B = self.spcreator([[1, 2], [3, 4], [5, 6]])
        assert_raises(ValueError, A.__matmul__, B)
        if isinstance(A, sparray):
            assert_raises(ValueError, A.__mul__, B)

    def test_matmat_dense(self):
        a = matrix([[3, 0, 0], [0, 1, 0], [2, 0, 3.0], [2, 3, 0]])
        asp = self.spcreator(a)
        bs = [array([[1, 2], [3, 4], [5, 6]]), matrix([[1, 2], [3, 4], [5, 6]])]
        for b in bs:
            result = asp @ b
            assert_(isinstance(result, type(b)))
            assert_equal(result.shape, (4, 2))
            assert_equal(result, dot(a, b))

    def test_sparse_format_conversions(self):
        A = sparse.kron([[1, 0, 2], [0, 3, 4], [5, 0, 0]], [[1, 2], [0, 3]])
        D = A.toarray()
        A = self.spcreator(A)
        for format in ['bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil']:
            a = A.asformat(format)
            assert_equal(a.format, format)
            assert_array_equal(a.toarray(), D)
            b = self.spcreator(D + 3j).asformat(format)
            assert_equal(b.format, format)
            assert_array_equal(b.toarray(), D + 3j)
            c = eval(format + '_matrix')(A)
            assert_equal(c.format, format)
            assert_array_equal(c.toarray(), D)
        for format in ['array', 'dense']:
            a = A.asformat(format)
            assert_array_equal(a, D)
            b = self.spcreator(D + 3j).asformat(format)
            assert_array_equal(b, D + 3j)

    def test_tobsr(self):
        x = array([[1, 0, 2, 0], [0, 0, 0, 0], [0, 0, 4, 5]])
        y = array([[0, 1, 2], [3, 0, 5]])
        A = kron(x, y)
        Asp = self.spcreator(A)
        for format in ['bsr']:
            fn = getattr(Asp, 'to' + format)
            for X in [1, 2, 3, 6]:
                for Y in [1, 2, 3, 4, 6, 12]:
                    assert_equal(fn(blocksize=(X, Y)).toarray(), A)

    def test_transpose(self):
        dat_1 = self.dat
        dat_2 = np.array([[]])
        matrices = [dat_1, dat_2]

        def check(dtype, j):
            dat = array(matrices[j], dtype=dtype)
            datsp = self.spcreator(dat)
            a = datsp.transpose()
            b = dat.transpose()
            assert_array_equal(a.toarray(), b)
            assert_array_equal(a.transpose().toarray(), dat)
            assert_array_equal(datsp.transpose(axes=(1, 0)).toarray(), b)
            assert_equal(a.dtype, b.dtype)
        empty = self.spcreator((3, 4))
        assert_array_equal(np.transpose(empty).toarray(), np.transpose(zeros((3, 4))))
        assert_array_equal(empty.T.toarray(), zeros((4, 3)))
        assert_raises(ValueError, empty.transpose, axes=0)
        for dtype in self.checked_dtypes:
            for j in range(len(matrices)):
                check(dtype, j)

    def test_add_dense(self):

        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            sum1 = dat + datsp
            assert_array_equal(sum1, dat + dat)
            sum2 = datsp + dat
            assert_array_equal(sum2, dat + dat)
        for dtype in self.math_dtypes:
            check(dtype)

    def test_sub_dense(self):

        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            if dat.dtype == bool:
                sum1 = dat - datsp
                assert_array_equal(sum1, dat - dat)
                sum2 = datsp - dat
                assert_array_equal(sum2, dat - dat)
            else:
                sum1 = dat + dat + dat - datsp
                assert_array_equal(sum1, dat + dat)
                sum2 = datsp + datsp + datsp - dat
                assert_array_equal(sum2, dat + dat)
        for dtype in self.math_dtypes:
            if dtype == np.dtype('bool'):
                continue
            check(dtype)

    def test_maximum_minimum(self):
        A_dense = np.array([[1, 0, 3], [0, 4, 5], [0, 0, 0]])
        B_dense = np.array([[1, 1, 2], [0, 3, 6], [1, -1, 0]])
        A_dense_cpx = np.array([[1, 0, 3], [0, 4 + 2j, 5], [0, 1j, -1j]])

        def check(dtype, dtype2, btype):
            if np.issubdtype(dtype, np.complexfloating):
                A = self.spcreator(A_dense_cpx.astype(dtype))
            else:
                A = self.spcreator(A_dense.astype(dtype))
            if btype == 'scalar':
                B = dtype2.type(1)
            elif btype == 'scalar2':
                B = dtype2.type(-1)
            elif btype == 'dense':
                B = B_dense.astype(dtype2)
            elif btype == 'sparse':
                B = self.spcreator(B_dense.astype(dtype2))
            else:
                raise ValueError()
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, 'Taking maximum .minimum. with > 0 .< 0. number results to a dense matrix')
                max_s = A.maximum(B)
                min_s = A.minimum(B)
            max_d = np.maximum(toarray(A), toarray(B))
            assert_array_equal(toarray(max_s), max_d)
            assert_equal(max_s.dtype, max_d.dtype)
            min_d = np.minimum(toarray(A), toarray(B))
            assert_array_equal(toarray(min_s), min_d)
            assert_equal(min_s.dtype, min_d.dtype)
        for dtype in self.math_dtypes:
            for dtype2 in [np.int8, np.float64, np.complex128]:
                for btype in ['scalar', 'scalar2', 'dense', 'sparse']:
                    check(np.dtype(dtype), np.dtype(dtype2), btype)

    def test_copy(self):
        A = self.datsp
        assert_equal(A.copy().format, A.format)
        assert_equal(A.__class__(A, copy=True).format, A.format)
        assert_equal(A.__class__(A, copy=False).format, A.format)
        assert_equal(A.copy().toarray(), A.toarray())
        assert_equal(A.__class__(A, copy=True).toarray(), A.toarray())
        assert_equal(A.__class__(A, copy=False).toarray(), A.toarray())
        toself = getattr(A, 'to' + A.format)
        assert_(toself() is A)
        assert_(toself(copy=False) is A)
        assert_equal(toself(copy=True).format, A.format)
        assert_equal(toself(copy=True).toarray(), A.toarray())
        assert_(not sparse_may_share_memory(A.copy(), A))

    def test_iterator(self):
        B = matrix(np.arange(50).reshape(5, 10))
        A = self.spcreator(B)
        for x, y in zip(A, B):
            assert_equal(x.toarray(), y)

    def test_size_zero_matrix_arithmetic(self):
        mat = array([])
        a = mat.reshape((0, 0))
        b = mat.reshape((0, 1))
        c = mat.reshape((0, 5))
        d = mat.reshape((1, 0))
        e = mat.reshape((5, 0))
        f = np.ones([5, 5])
        asp = self.spcreator(a)
        bsp = self.spcreator(b)
        csp = self.spcreator(c)
        dsp = self.spcreator(d)
        esp = self.spcreator(e)
        fsp = self.spcreator(f)
        assert_array_equal(asp.dot(asp).toarray(), np.dot(a, a))
        assert_array_equal(bsp.dot(dsp).toarray(), np.dot(b, d))
        assert_array_equal(dsp.dot(bsp).toarray(), np.dot(d, b))
        assert_array_equal(csp.dot(esp).toarray(), np.dot(c, e))
        assert_array_equal(csp.dot(fsp).toarray(), np.dot(c, f))
        assert_array_equal(esp.dot(csp).toarray(), np.dot(e, c))
        assert_array_equal(dsp.dot(csp).toarray(), np.dot(d, c))
        assert_array_equal(fsp.dot(esp).toarray(), np.dot(f, e))
        assert_raises(ValueError, dsp.dot, e)
        assert_raises(ValueError, asp.dot, d)
        assert_array_equal(asp.multiply(asp).toarray(), np.multiply(a, a))
        assert_array_equal(bsp.multiply(bsp).toarray(), np.multiply(b, b))
        assert_array_equal(dsp.multiply(dsp).toarray(), np.multiply(d, d))
        assert_array_equal(asp.multiply(a).toarray(), np.multiply(a, a))
        assert_array_equal(bsp.multiply(b).toarray(), np.multiply(b, b))
        assert_array_equal(dsp.multiply(d).toarray(), np.multiply(d, d))
        assert_array_equal(asp.multiply(6).toarray(), np.multiply(a, 6))
        assert_array_equal(bsp.multiply(6).toarray(), np.multiply(b, 6))
        assert_array_equal(dsp.multiply(6).toarray(), np.multiply(d, 6))
        assert_raises(ValueError, asp.multiply, c)
        assert_raises(ValueError, esp.multiply, c)
        assert_array_equal(asp.__add__(asp).toarray(), a.__add__(a))
        assert_array_equal(bsp.__add__(bsp).toarray(), b.__add__(b))
        assert_array_equal(dsp.__add__(dsp).toarray(), d.__add__(d))
        assert_raises(ValueError, asp.__add__, dsp)
        assert_raises(ValueError, bsp.__add__, asp)

    def test_size_zero_conversions(self):
        mat = array([])
        a = mat.reshape((0, 0))
        b = mat.reshape((0, 5))
        c = mat.reshape((5, 0))
        for m in [a, b, c]:
            spm = self.spcreator(m)
            assert_array_equal(spm.tocoo().toarray(), m)
            assert_array_equal(spm.tocsr().toarray(), m)
            assert_array_equal(spm.tocsc().toarray(), m)
            assert_array_equal(spm.tolil().toarray(), m)
            assert_array_equal(spm.todok().toarray(), m)
            assert_array_equal(spm.tobsr().toarray(), m)

    def test_pickle(self):
        import pickle
        sup = suppress_warnings()
        sup.filter(SparseEfficiencyWarning)

        @sup
        def check():
            datsp = self.datsp.copy()
            for protocol in range(pickle.HIGHEST_PROTOCOL):
                sploaded = pickle.loads(pickle.dumps(datsp, protocol=protocol))
                assert_equal(datsp.shape, sploaded.shape)
                assert_array_equal(datsp.toarray(), sploaded.toarray())
                assert_equal(datsp.format, sploaded.format)
                for key, val in datsp.__dict__.items():
                    if isinstance(val, np.ndarray):
                        assert_array_equal(val, sploaded.__dict__[key])
                    else:
                        assert_(val == sploaded.__dict__[key])
        check()

    def test_unary_ufunc_overrides(self):

        def check(name):
            if name == 'sign':
                pytest.skip('sign conflicts with comparison op support on Numpy')
            if self.spcreator in (dok_matrix, lil_matrix):
                pytest.skip('Unary ops not implemented for dok/lil')
            ufunc = getattr(np, name)
            X = self.spcreator(np.arange(20).reshape(4, 5) / 20.0)
            X0 = ufunc(X.toarray())
            X2 = ufunc(X)
            assert_array_equal(X2.toarray(), X0)
        for name in ['sin', 'tan', 'arcsin', 'arctan', 'sinh', 'tanh', 'arcsinh', 'arctanh', 'rint', 'sign', 'expm1', 'log1p', 'deg2rad', 'rad2deg', 'floor', 'ceil', 'trunc', 'sqrt', 'abs']:
            check(name)

    def test_resize(self):
        D = np.array([[1, 0, 3, 4], [2, 0, 0, 0], [3, 0, 0, 0]])
        S = self.spcreator(D)
        assert_(S.resize((3, 2)) is None)
        assert_array_equal(S.toarray(), [[1, 0], [2, 0], [3, 0]])
        S.resize((2, 2))
        assert_array_equal(S.toarray(), [[1, 0], [2, 0]])
        S.resize((3, 2))
        assert_array_equal(S.toarray(), [[1, 0], [2, 0], [0, 0]])
        S.resize((3, 3))
        assert_array_equal(S.toarray(), [[1, 0, 0], [2, 0, 0], [0, 0, 0]])
        S.resize((3, 3))
        assert_array_equal(S.toarray(), [[1, 0, 0], [2, 0, 0], [0, 0, 0]])
        S.resize(3, 2)
        assert_array_equal(S.toarray(), [[1, 0], [2, 0], [0, 0]])
        for bad_shape in [1, (-1, 2), (2, -1), (1, 2, 3)]:
            assert_raises(ValueError, S.resize, bad_shape)

    def test_constructor1_base(self):
        A = self.datsp
        self_format = A.format
        C = A.__class__(A, copy=False)
        assert_array_equal_dtype(A.toarray(), C.toarray())
        if self_format not in NON_ARRAY_BACKED_FORMATS:
            assert_(sparse_may_share_memory(A, C))
        C = A.__class__(A, dtype=A.dtype, copy=False)
        assert_array_equal_dtype(A.toarray(), C.toarray())
        if self_format not in NON_ARRAY_BACKED_FORMATS:
            assert_(sparse_may_share_memory(A, C))
        C = A.__class__(A, dtype=np.float32, copy=False)
        assert_array_equal(A.toarray(), C.toarray())
        C = A.__class__(A, copy=True)
        assert_array_equal_dtype(A.toarray(), C.toarray())
        assert_(not sparse_may_share_memory(A, C))
        for other_format in ['csr', 'csc', 'coo', 'dia', 'dok', 'lil']:
            if other_format == self_format:
                continue
            B = A.asformat(other_format)
            C = A.__class__(B, copy=False)
            assert_array_equal_dtype(A.toarray(), C.toarray())
            C = A.__class__(B, copy=True)
            assert_array_equal_dtype(A.toarray(), C.toarray())
            assert_(not sparse_may_share_memory(B, C))