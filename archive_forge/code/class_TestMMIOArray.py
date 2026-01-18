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
class TestMMIOArray:

    def setup_method(self):
        self.tmpdir = mkdtemp()
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def check(self, a, info):
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_array_almost_equal(a, b)

    def check_exact(self, a, info):
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_equal(a, b)

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_integer(self, typeval, dtype):
        self.check_exact(array([[1, 2], [3, 4]], dtype=dtype), (2, 2, 4, 'array', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_32bit_integer(self, typeval, dtype):
        a = array([[2 ** 31 - 1, 2 ** 31 - 2], [2 ** 31 - 3, 2 ** 31 - 4]], dtype=dtype)
        self.check_exact(a, (2, 2, 4, 'array', typeval, 'general'))

    def test_64bit_integer(self):
        a = array([[2 ** 31, 2 ** 32], [2 ** 63 - 2, 2 ** 63 - 1]], dtype=np.int64)
        if np.intp(0).itemsize < 8 and mmwrite == scipy.io._mmio.mmwrite:
            assert_raises(OverflowError, mmwrite, self.fn, a)
        else:
            self.check_exact(a, (2, 2, 4, 'array', 'integer', 'general'))

    def test_64bit_unsigned_integer(self):
        a = array([[2 ** 31, 2 ** 32], [2 ** 64 - 2, 2 ** 64 - 1]], dtype=np.uint64)
        self.check_exact(a, (2, 2, 4, 'array', 'unsigned-integer', 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_upper_triangle_integer(self, typeval, dtype):
        self.check_exact(array([[0, 1], [0, 0]], dtype=dtype), (2, 2, 4, 'array', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_lower_triangle_integer(self, typeval, dtype):
        self.check_exact(array([[0, 0], [1, 0]], dtype=dtype), (2, 2, 4, 'array', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_rectangular_integer(self, typeval, dtype):
        self.check_exact(array([[1, 2, 3], [4, 5, 6]], dtype=dtype), (2, 3, 6, 'array', typeval, 'general'))

    def test_simple_rectangular_float(self):
        self.check([[1, 2], [3.5, 4], [5, 6]], (3, 2, 6, 'array', 'real', 'general'))

    def test_simple_float(self):
        self.check([[1, 2], [3, 4.0]], (2, 2, 4, 'array', 'real', 'general'))

    def test_simple_complex(self):
        self.check([[1, 2], [3, 4j]], (2, 2, 4, 'array', 'complex', 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_symmetric_integer(self, typeval, dtype):
        self.check_exact(array([[1, 2], [2, 4]], dtype=dtype), (2, 2, 4, 'array', typeval, 'symmetric'))

    def test_simple_skew_symmetric_integer(self):
        self.check_exact([[0, 2], [-2, 0]], (2, 2, 4, 'array', 'integer', 'skew-symmetric'))

    def test_simple_skew_symmetric_float(self):
        self.check(array([[0, 2], [-2.0, 0.0]], 'f'), (2, 2, 4, 'array', 'real', 'skew-symmetric'))

    def test_simple_hermitian_complex(self):
        self.check([[1, 2 + 3j], [2 - 3j, 4]], (2, 2, 4, 'array', 'complex', 'hermitian'))

    def test_random_symmetric_float(self):
        sz = (20, 20)
        a = np.random.random(sz)
        a = a + transpose(a)
        self.check(a, (20, 20, 400, 'array', 'real', 'symmetric'))

    def test_random_rectangular_float(self):
        sz = (20, 15)
        a = np.random.random(sz)
        self.check(a, (20, 15, 300, 'array', 'real', 'general'))

    def test_bad_number_of_array_header_fields(self):
        s = '            %%MatrixMarket matrix array real general\n              3  3 999\n            1.0\n            2.0\n            3.0\n            4.0\n            5.0\n            6.0\n            7.0\n            8.0\n            9.0\n            '
        text = textwrap.dedent(s).encode('ascii')
        with pytest.raises(ValueError, match='not of length 2'):
            scipy.io.mmread(io.BytesIO(text))

    def test_gh13634_non_skew_symmetric_int(self):
        self.check_exact(array([[1, 2], [-2, 99]], dtype=np.int32), (2, 2, 4, 'array', 'integer', 'general'))

    def test_gh13634_non_skew_symmetric_float(self):
        self.check(array([[1, 2], [-2, 99.0]], dtype=np.float32), (2, 2, 4, 'array', 'real', 'general'))