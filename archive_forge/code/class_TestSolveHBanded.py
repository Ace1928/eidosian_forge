import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
class TestSolveHBanded:

    def test_01_upper(self):
        ab = array([[0.0, 0.0, 2.0, 2.0], [-99, 1.0, 1.0, 1.0], [4.0, 4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0, 2.0])
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0, 0.0])

    def test_02_upper(self):
        ab = array([[0.0, 0.0, 2.0, 2.0], [-99, 1.0, 1.0, 1.0], [4.0, 4.0, 4.0, 4.0]])
        b = array([[1.0, 6.0], [4.0, 2.0], [1.0, 6.0], [2.0, 1.0]])
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        assert_array_almost_equal(x, expected)

    def test_03_upper(self):
        ab = array([[0.0, 0.0, 2.0, 2.0], [-99, 1.0, 1.0, 1.0], [4.0, 4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0, 2.0]).reshape(-1, 1)
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, array([0.0, 1.0, 0.0, 0.0]).reshape(-1, 1))

    def test_01_lower(self):
        ab = array([[4.0, 4.0, 4.0, 4.0], [1.0, 1.0, 1.0, -99], [2.0, 2.0, 0.0, 0.0]])
        b = array([1.0, 4.0, 1.0, 2.0])
        x = solveh_banded(ab, b, lower=True)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0, 0.0])

    def test_02_lower(self):
        ab = array([[4.0, 4.0, 4.0, 4.0], [1.0, 1.0, 1.0, -99], [2.0, 2.0, 0.0, 0.0]])
        b = array([[1.0, 6.0], [4.0, 2.0], [1.0, 6.0], [2.0, 1.0]])
        x = solveh_banded(ab, b, lower=True)
        expected = array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        assert_array_almost_equal(x, expected)

    def test_01_float32(self):
        ab = array([[0.0, 0.0, 2.0, 2.0], [-99, 1.0, 1.0, 1.0], [4.0, 4.0, 4.0, 4.0]], dtype=float32)
        b = array([1.0, 4.0, 1.0, 2.0], dtype=float32)
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0, 0.0])

    def test_02_float32(self):
        ab = array([[0.0, 0.0, 2.0, 2.0], [-99, 1.0, 1.0, 1.0], [4.0, 4.0, 4.0, 4.0]], dtype=float32)
        b = array([[1.0, 6.0], [4.0, 2.0], [1.0, 6.0], [2.0, 1.0]], dtype=float32)
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        assert_array_almost_equal(x, expected)

    def test_01_complex(self):
        ab = array([[0.0, 0.0, 2.0, 2.0], [-99, -1j, -1j, -1j], [4.0, 4.0, 4.0, 4.0]])
        b = array([2 - 1j, 4.0 - 1j, 4 + 1j, 2 + 1j])
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 1.0, 0.0])

    def test_02_complex(self):
        ab = array([[0.0, 0.0, 2.0, 2.0], [-99, -1j, -1j, -1j], [4.0, 4.0, 4.0, 4.0]])
        b = array([[2 - 1j, 2 + 4j], [4.0 - 1j, -1 - 1j], [4.0 + 1j, 4 + 2j], [2 + 1j, 1j]])
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1j], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        assert_array_almost_equal(x, expected)

    def test_tridiag_01_upper(self):
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0])
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0])

    def test_tridiag_02_upper(self):
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]])
        b = array([[1.0, 4.0], [4.0, 2.0], [1.0, 4.0]])
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        assert_array_almost_equal(x, expected)

    def test_tridiag_03_upper(self):
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0]).reshape(-1, 1)
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, array([0.0, 1.0, 0.0]).reshape(-1, 1))

    def test_tridiag_01_lower(self):
        ab = array([[4.0, 4.0, 4.0], [1.0, 1.0, -99]])
        b = array([1.0, 4.0, 1.0])
        x = solveh_banded(ab, b, lower=True)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0])

    def test_tridiag_02_lower(self):
        ab = array([[4.0, 4.0, 4.0], [1.0, 1.0, -99]])
        b = array([[1.0, 4.0], [4.0, 2.0], [1.0, 4.0]])
        x = solveh_banded(ab, b, lower=True)
        expected = array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        assert_array_almost_equal(x, expected)

    def test_tridiag_01_float32(self):
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]], dtype=float32)
        b = array([1.0, 4.0, 1.0], dtype=float32)
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0])

    def test_tridiag_02_float32(self):
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]], dtype=float32)
        b = array([[1.0, 4.0], [4.0, 2.0], [1.0, 4.0]], dtype=float32)
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        assert_array_almost_equal(x, expected)

    def test_tridiag_01_complex(self):
        ab = array([[-99, -1j, -1j], [4.0, 4.0, 4.0]])
        b = array([-1j, 4.0 - 1j, 4 + 1j])
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 1.0])

    def test_tridiag_02_complex(self):
        ab = array([[-99, -1j, -1j], [4.0, 4.0, 4.0]])
        b = array([[-1j, 4j], [4.0 - 1j, -1.0 - 1j], [4.0 + 1j, 4.0]])
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1j], [1.0, 0.0], [1.0, 1.0]])
        assert_array_almost_equal(x, expected)

    def test_check_finite(self):
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0])
        x = solveh_banded(ab, b, check_finite=False)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0])

    def test_bad_shapes(self):
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]])
        b = array([[1.0, 4.0], [4.0, 2.0]])
        assert_raises(ValueError, solveh_banded, ab, b)
        assert_raises(ValueError, solveh_banded, ab, [1.0, 2.0])
        assert_raises(ValueError, solveh_banded, ab, [1.0])

    def test_1x1(self):
        x = solveh_banded([[1]], [[1, 2, 3]])
        assert_array_equal(x, [[1.0, 2.0, 3.0]])
        assert_equal(x.dtype, np.dtype('f8'))

    def test_native_list_arguments(self):
        ab = [[0.0, 0.0, 2.0, 2.0], [-99, 1.0, 1.0, 1.0], [4.0, 4.0, 4.0, 4.0]]
        b = [1.0, 4.0, 1.0, 2.0]
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0, 0.0])