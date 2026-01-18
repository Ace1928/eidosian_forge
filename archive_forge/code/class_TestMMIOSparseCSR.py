from tempfile import mkdtemp
import os
import io
import shutil
import textwrap
import numpy as np
from numpy import array, transpose, pi
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
import scipy.sparse
import scipy.io._mmio
import scipy.io._fast_matrix_market as fmm
class TestMMIOSparseCSR(TestMMIOArray):

    def setup_method(self):
        self.tmpdir = mkdtemp()
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def check(self, a, info):
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_array_almost_equal(a.toarray(), b.toarray())

    def check_exact(self, a, info):
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_equal(a.toarray(), b.toarray())

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_integer(self, typeval, dtype):
        self.check_exact(scipy.sparse.csr_matrix([[1, 2], [3, 4]], dtype=dtype), (2, 2, 4, 'coordinate', typeval, 'general'))

    def test_32bit_integer(self):
        a = scipy.sparse.csr_matrix(array([[2 ** 31 - 1, -2 ** 31 + 2], [2 ** 31 - 3, 2 ** 31 - 4]], dtype=np.int32))
        self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))

    def test_64bit_integer(self):
        a = scipy.sparse.csr_matrix(array([[2 ** 32 + 1, 2 ** 32 + 1], [-2 ** 63 + 2, 2 ** 63 - 2]], dtype=np.int64))
        if np.intp(0).itemsize < 8 and mmwrite == scipy.io._mmio.mmwrite:
            assert_raises(OverflowError, mmwrite, self.fn, a)
        else:
            self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))

    def test_32bit_unsigned_integer(self):
        a = scipy.sparse.csr_matrix(array([[2 ** 31 - 1, 2 ** 31 - 2], [2 ** 31 - 3, 2 ** 31 - 4]], dtype=np.uint32))
        self.check_exact(a, (2, 2, 4, 'coordinate', 'unsigned-integer', 'general'))

    def test_64bit_unsigned_integer(self):
        a = scipy.sparse.csr_matrix(array([[2 ** 32 + 1, 2 ** 32 + 1], [2 ** 64 - 2, 2 ** 64 - 1]], dtype=np.uint64))
        self.check_exact(a, (2, 2, 4, 'coordinate', 'unsigned-integer', 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_upper_triangle_integer(self, typeval, dtype):
        self.check_exact(scipy.sparse.csr_matrix([[0, 1], [0, 0]], dtype=dtype), (2, 2, 1, 'coordinate', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_lower_triangle_integer(self, typeval, dtype):
        self.check_exact(scipy.sparse.csr_matrix([[0, 0], [1, 0]], dtype=dtype), (2, 2, 1, 'coordinate', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_rectangular_integer(self, typeval, dtype):
        self.check_exact(scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6]], dtype=dtype), (2, 3, 6, 'coordinate', typeval, 'general'))

    def test_simple_rectangular_float(self):
        self.check(scipy.sparse.csr_matrix([[1, 2], [3.5, 4], [5, 6]]), (3, 2, 6, 'coordinate', 'real', 'general'))

    def test_simple_float(self):
        self.check(scipy.sparse.csr_matrix([[1, 2], [3, 4.0]]), (2, 2, 4, 'coordinate', 'real', 'general'))

    def test_simple_complex(self):
        self.check(scipy.sparse.csr_matrix([[1, 2], [3, 4j]]), (2, 2, 4, 'coordinate', 'complex', 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_symmetric_integer(self, typeval, dtype):
        self.check_exact(scipy.sparse.csr_matrix([[1, 2], [2, 4]], dtype=dtype), (2, 2, 3, 'coordinate', typeval, 'symmetric'))

    def test_simple_skew_symmetric_integer(self):
        self.check_exact(scipy.sparse.csr_matrix([[0, 2], [-2, 0]]), (2, 2, 1, 'coordinate', 'integer', 'skew-symmetric'))

    def test_simple_skew_symmetric_float(self):
        self.check(scipy.sparse.csr_matrix(array([[0, 2], [-2.0, 0]], 'f')), (2, 2, 1, 'coordinate', 'real', 'skew-symmetric'))

    def test_simple_hermitian_complex(self):
        self.check(scipy.sparse.csr_matrix([[1, 2 + 3j], [2 - 3j, 4]]), (2, 2, 3, 'coordinate', 'complex', 'hermitian'))

    def test_random_symmetric_float(self):
        sz = (20, 20)
        a = np.random.random(sz)
        a = a + transpose(a)
        a = scipy.sparse.csr_matrix(a)
        self.check(a, (20, 20, 210, 'coordinate', 'real', 'symmetric'))

    def test_random_rectangular_float(self):
        sz = (20, 15)
        a = np.random.random(sz)
        a = scipy.sparse.csr_matrix(a)
        self.check(a, (20, 15, 300, 'coordinate', 'real', 'general'))

    def test_simple_pattern(self):
        a = scipy.sparse.csr_matrix([[0, 1.5], [3.0, 2.5]])
        p = np.zeros_like(a.toarray())
        p[a.toarray() > 0] = 1
        info = (2, 2, 3, 'coordinate', 'pattern', 'general')
        mmwrite(self.fn, a, field='pattern')
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_array_almost_equal(p, b.toarray())

    def test_gh13634_non_skew_symmetric_int(self):
        a = scipy.sparse.csr_matrix([[1, 2], [-2, 99]], dtype=np.int32)
        self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))

    def test_gh13634_non_skew_symmetric_float(self):
        a = scipy.sparse.csr_matrix([[1, 2], [-2, 99.0]], dtype=np.float32)
        self.check(a, (2, 2, 4, 'coordinate', 'real', 'general'))