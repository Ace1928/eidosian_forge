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
class TestPinv:

    def setup_method(self):
        np.random.seed(1234)

    def test_simple_real(self):
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
        a_pinv = pinv(a)
        assert_array_almost_equal(dot(a, a_pinv), np.eye(3))

    def test_simple_complex(self):
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float) + 1j * array([[10, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=float)
        a_pinv = pinv(a)
        assert_array_almost_equal(dot(a, a_pinv), np.eye(3))

    def test_simple_singular(self):
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        a_pinv = pinv(a)
        expected = array([[-0.638888889, -0.166666667, 0.305555556], [-0.0555555556, 1.30136518e-16, 0.0555555556], [0.527777778, 0.166666667, -0.194444444]])
        assert_array_almost_equal(a_pinv, expected)

    def test_simple_cols(self):
        a = array([[1, 2, 3], [4, 5, 6]], dtype=float)
        a_pinv = pinv(a)
        expected = array([[-0.94444444, 0.44444444], [-0.11111111, 0.11111111], [0.72222222, -0.22222222]])
        assert_array_almost_equal(a_pinv, expected)

    def test_simple_rows(self):
        a = array([[1, 2], [3, 4], [5, 6]], dtype=float)
        a_pinv = pinv(a)
        expected = array([[-1.33333333, -0.33333333, 0.66666667], [1.08333333, 0.33333333, -0.41666667]])
        assert_array_almost_equal(a_pinv, expected)

    def test_check_finite(self):
        a = array([[1, 2, 3], [4, 5, 6.0], [7, 8, 10]])
        a_pinv = pinv(a, check_finite=False)
        assert_array_almost_equal(dot(a, a_pinv), np.eye(3))

    def test_native_list_argument(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        a_pinv = pinv(a)
        expected = array([[-0.638888889, -0.166666667, 0.305555556], [-0.0555555556, 1.30136518e-16, 0.0555555556], [0.527777778, 0.166666667, -0.194444444]])
        assert_array_almost_equal(a_pinv, expected)

    def test_atol_rtol(self):
        n = 12
        q, _ = qr(np.random.rand(n, n))
        a_m = np.arange(35.0).reshape(7, 5)
        a = a_m.copy()
        a[0, 0] = 0.001
        atol = 1e-05
        rtol = 0.05
        a_p = pinv(a_m, atol=atol, rtol=0.0)
        adiff1 = a @ a_p @ a - a
        adiff2 = a_m @ a_p @ a_m - a_m
        assert_allclose(np.linalg.norm(adiff1), 0.0005, atol=0.0005)
        assert_allclose(np.linalg.norm(adiff2), 5e-14, atol=5e-14)
        a_p = pinv(a_m, atol=atol, rtol=rtol)
        adiff1 = a @ a_p @ a - a
        adiff2 = a_m @ a_p @ a_m - a_m
        assert_allclose(np.linalg.norm(adiff1), 4.233, rtol=0.01)
        assert_allclose(np.linalg.norm(adiff2), 4.233, rtol=0.01)

    @pytest.mark.parametrize('cond', [1, None, _NoValue])
    @pytest.mark.parametrize('rcond', [1, None, _NoValue])
    def test_cond_rcond_deprecation(self, cond, rcond):
        if cond is _NoValue and rcond is _NoValue:
            pinv(np.ones((2, 2)), cond=cond, rcond=rcond)
        else:
            with pytest.deprecated_call(match='"cond" and "rcond"'):
                pinv(np.ones((2, 2)), cond=cond, rcond=rcond)

    def test_positional_deprecation(self):
        with pytest.deprecated_call(match='use keyword arguments'):
            pinv(np.ones((2, 2)), 0.0, 1e-10)