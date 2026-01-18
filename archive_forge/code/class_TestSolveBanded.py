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
class TestSolveBanded:

    def test_real(self):
        a = array([[1.0, 20, 0, 0], [-30, 4, 6, 0], [2, 1, 20, 2], [0, -1, 7, 14]])
        ab = array([[0.0, 20, 6, 2], [1, 4, 20, 14], [-30, 1, 7, 0], [2, -1, 0, 0]])
        l, u = (2, 1)
        b4 = array([10.0, 0.0, 2.0, 14.0])
        b4by1 = b4.reshape(-1, 1)
        b4by2 = array([[2, 1], [-30, 4], [2, 3], [1, 3]])
        b4by4 = array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0]])
        for b in [b4, b4by1, b4by2, b4by4]:
            x = solve_banded((l, u), ab, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_complex(self):
        a = array([[1.0, 20, 0, 0], [-30, 4, 6, 0], [2j, 1, 20, 2j], [0, -1, 7, 14]])
        ab = array([[0.0, 20, 6, 2j], [1, 4, 20, 14], [-30, 1, 7, 0], [2j, -1, 0, 0]])
        l, u = (2, 1)
        b4 = array([10.0, 0.0, 2.0, 14j])
        b4by1 = b4.reshape(-1, 1)
        b4by2 = array([[2, 1], [-30, 4], [2, 3], [1, 3]])
        b4by4 = array([[1, 0, 0, 0], [0, 0, 0, 1j], [0, 1, 0, 0], [0, 1, 0, 0]])
        for b in [b4, b4by1, b4by2, b4by4]:
            x = solve_banded((l, u), ab, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_tridiag_real(self):
        ab = array([[0.0, 20, 6, 2], [1, 4, 20, 14], [-30, 1, 7, 0]])
        a = np.diag(ab[0, 1:], 1) + np.diag(ab[1, :], 0) + np.diag(ab[2, :-1], -1)
        b4 = array([10.0, 0.0, 2.0, 14.0])
        b4by1 = b4.reshape(-1, 1)
        b4by2 = array([[2, 1], [-30, 4], [2, 3], [1, 3]])
        b4by4 = array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0]])
        for b in [b4, b4by1, b4by2, b4by4]:
            x = solve_banded((1, 1), ab, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_tridiag_complex(self):
        ab = array([[0.0, 20, 6, 2j], [1, 4, 20, 14], [-30, 1, 7, 0]])
        a = np.diag(ab[0, 1:], 1) + np.diag(ab[1, :], 0) + np.diag(ab[2, :-1], -1)
        b4 = array([10.0, 0.0, 2.0, 14j])
        b4by1 = b4.reshape(-1, 1)
        b4by2 = array([[2, 1], [-30, 4], [2, 3], [1, 3]])
        b4by4 = array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0]])
        for b in [b4, b4by1, b4by2, b4by4]:
            x = solve_banded((1, 1), ab, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_check_finite(self):
        a = array([[1.0, 20, 0, 0], [-30, 4, 6, 0], [2, 1, 20, 2], [0, -1, 7, 14]])
        ab = array([[0.0, 20, 6, 2], [1, 4, 20, 14], [-30, 1, 7, 0], [2, -1, 0, 0]])
        l, u = (2, 1)
        b4 = array([10.0, 0.0, 2.0, 14.0])
        x = solve_banded((l, u), ab, b4, check_finite=False)
        assert_array_almost_equal(dot(a, x), b4)

    def test_bad_shape(self):
        ab = array([[0.0, 20, 6, 2], [1, 4, 20, 14], [-30, 1, 7, 0], [2, -1, 0, 0]])
        l, u = (2, 1)
        bad = array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 4)
        assert_raises(ValueError, solve_banded, (l, u), ab, bad)
        assert_raises(ValueError, solve_banded, (l, u), ab, [1.0, 2.0])
        assert_raises(ValueError, solve_banded, (1, 1), ab, [1.0, 2.0])

    def test_1x1(self):
        b = array([[1.0, 2.0, 3.0]])
        x = solve_banded((1, 1), [[0], [2], [0]], b)
        assert_array_equal(x, [[0.5, 1.0, 1.5]])
        assert_equal(x.dtype, np.dtype('f8'))
        assert_array_equal(b, [[1.0, 2.0, 3.0]])

    def test_native_list_arguments(self):
        a = [[1.0, 20, 0, 0], [-30, 4, 6, 0], [2, 1, 20, 2], [0, -1, 7, 14]]
        ab = [[0.0, 20, 6, 2], [1, 4, 20, 14], [-30, 1, 7, 0], [2, -1, 0, 0]]
        l, u = (2, 1)
        b = [10.0, 0.0, 2.0, 14.0]
        x = solve_banded((l, u), ab, b)
        assert_array_almost_equal(dot(a, x), b)