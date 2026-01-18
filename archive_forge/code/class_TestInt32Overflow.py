import sys
import os
import gc
import threading
import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from scipy.sparse import (_sparsetools, coo_matrix, csr_matrix, csc_matrix,
from scipy.sparse._sputils import supported_dtypes
from scipy._lib._testutils import check_free_memory
import pytest
from pytest import raises as assert_raises
@pytest.mark.skipif(not (sys.platform.startswith('linux') and np.dtype(np.intp).itemsize >= 8), reason='test requires 64-bit Linux')
class TestInt32Overflow:
    """
    Some of the sparsetools routines use dense 2D matrices whose
    total size is not bounded by the nnz of the sparse matrix. These
    routines used to suffer from int32 wraparounds; here, we try to
    check that the wraparounds don't occur any more.
    """
    n = 50000

    def setup_method(self):
        assert self.n ** 2 > np.iinfo(np.int32).max
        try:
            parallel_count = int(os.environ.get('PYTEST_XDIST_WORKER_COUNT', '1'))
        except ValueError:
            parallel_count = np.inf
        check_free_memory(3000 * parallel_count)

    def teardown_method(self):
        gc.collect()

    def test_coo_todense(self):
        n = self.n
        i = np.array([0, n - 1])
        j = np.array([0, n - 1])
        data = np.array([1, 2], dtype=np.int8)
        m = coo_matrix((data, (i, j)))
        r = m.todense()
        assert_equal(r[0, 0], 1)
        assert_equal(r[-1, -1], 2)
        del r
        gc.collect()

    @pytest.mark.slow
    def test_matvecs(self):
        n = self.n
        i = np.array([0, n - 1])
        j = np.array([0, n - 1])
        data = np.array([1, 2], dtype=np.int8)
        m = coo_matrix((data, (i, j)))
        b = np.ones((n, n), dtype=np.int8)
        for sptype in (csr_matrix, csc_matrix, bsr_matrix):
            m2 = sptype(m)
            r = m2.dot(b)
            assert_equal(r[0, 0], 1)
            assert_equal(r[-1, -1], 2)
            del r
            gc.collect()
        del b
        gc.collect()

    @pytest.mark.slow
    def test_dia_matvec(self):
        n = self.n
        data = np.ones((n, n), dtype=np.int8)
        offsets = np.arange(n)
        m = dia_matrix((data, offsets), shape=(n, n))
        v = np.ones(m.shape[1], dtype=np.int8)
        r = m.dot(v)
        assert_equal(r[0], int_to_int8(n))
        del data, offsets, m, v, r
        gc.collect()
    _bsr_ops = [pytest.param('matmat', marks=pytest.mark.xslow), pytest.param('matvecs', marks=pytest.mark.xslow), 'matvec', 'diagonal', 'sort_indices', pytest.param('transpose', marks=pytest.mark.xslow)]

    @pytest.mark.slow
    @pytest.mark.parametrize('op', _bsr_ops)
    def test_bsr_1_block(self, op):

        def get_matrix():
            n = self.n
            data = np.ones((1, n, n), dtype=np.int8)
            indptr = np.array([0, 1], dtype=np.int32)
            indices = np.array([0], dtype=np.int32)
            m = bsr_matrix((data, indices, indptr), blocksize=(n, n), copy=False)
            del data, indptr, indices
            return m
        gc.collect()
        try:
            getattr(self, '_check_bsr_' + op)(get_matrix)
        finally:
            gc.collect()

    @pytest.mark.slow
    @pytest.mark.parametrize('op', _bsr_ops)
    def test_bsr_n_block(self, op):

        def get_matrix():
            n = self.n
            data = np.ones((n, n, 1), dtype=np.int8)
            indptr = np.array([0, n], dtype=np.int32)
            indices = np.arange(n, dtype=np.int32)
            m = bsr_matrix((data, indices, indptr), blocksize=(n, 1), copy=False)
            del data, indptr, indices
            return m
        gc.collect()
        try:
            getattr(self, '_check_bsr_' + op)(get_matrix)
        finally:
            gc.collect()

    def _check_bsr_matvecs(self, m):
        m = m()
        n = self.n
        r = m.dot(np.ones((n, 2), dtype=np.int8))
        assert_equal(r[0, 0], int_to_int8(n))

    def _check_bsr_matvec(self, m):
        m = m()
        n = self.n
        r = m.dot(np.ones((n,), dtype=np.int8))
        assert_equal(r[0], int_to_int8(n))

    def _check_bsr_diagonal(self, m):
        m = m()
        n = self.n
        r = m.diagonal()
        assert_equal(r, np.ones(n))

    def _check_bsr_sort_indices(self, m):
        m = m()
        m.sort_indices()

    def _check_bsr_transpose(self, m):
        m = m()
        m.transpose()

    def _check_bsr_matmat(self, m):
        m = m()
        n = self.n
        m2 = bsr_matrix(np.ones((n, 2), dtype=np.int8), blocksize=(m.blocksize[1], 2))
        m.dot(m2)
        del m2
        m2 = bsr_matrix(np.ones((2, n), dtype=np.int8), blocksize=(2, m.blocksize[0]))
        m2.dot(m)