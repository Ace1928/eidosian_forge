import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
class TestBlockDiag:

    def test_basic(self):
        x = block_diag(eye(2), [[1, 2], [3, 4], [5, 6]], [[1, 2, 3]])
        assert_array_equal(x, [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 2, 0, 0, 0], [0, 0, 3, 4, 0, 0, 0], [0, 0, 5, 6, 0, 0, 0], [0, 0, 0, 0, 1, 2, 3]])

    def test_dtype(self):
        x = block_diag([[1.5]])
        assert_equal(x.dtype, float)
        x = block_diag([[True]])
        assert_equal(x.dtype, bool)

    def test_mixed_dtypes(self):
        actual = block_diag([[1]], [[1j]])
        desired = np.array([[1, 0], [0, 1j]])
        assert_array_equal(actual, desired)

    def test_scalar_and_1d_args(self):
        a = block_diag(1)
        assert_equal(a.shape, (1, 1))
        assert_array_equal(a, [[1]])
        a = block_diag([2, 3], 4)
        assert_array_equal(a, [[2, 3, 0], [0, 0, 4]])

    def test_bad_arg(self):
        assert_raises(ValueError, block_diag, [[[1]]])

    def test_no_args(self):
        a = block_diag()
        assert_equal(a.ndim, 2)
        assert_equal(a.nbytes, 0)

    def test_empty_matrix_arg(self):
        a = block_diag([[1, 0], [0, 1]], [], [[2, 3], [4, 5], [6, 7]])
        assert_array_equal(a, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 2, 3], [0, 0, 4, 5], [0, 0, 6, 7]])

    def test_zerosized_matrix_arg(self):
        a = block_diag([[1, 0], [0, 1]], [[]], [[2, 3], [4, 5], [6, 7]], np.zeros([0, 2], dtype='int32'))
        assert_array_equal(a, [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 2, 3, 0, 0], [0, 0, 4, 5, 0, 0], [0, 0, 6, 7, 0, 0]])