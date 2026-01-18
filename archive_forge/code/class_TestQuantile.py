import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
class TestQuantile:

    def V(self, x, y, alpha):
        return (x >= y) - alpha

    def test_max_ulp(self):
        x = [0.0, 0.2, 0.4]
        a = np.quantile(x, 0.45)
        np.testing.assert_array_max_ulp(a, 0.18, maxulp=1)

    def test_basic(self):
        x = np.arange(8) * 0.5
        assert_equal(np.quantile(x, 0), 0.0)
        assert_equal(np.quantile(x, 1), 3.5)
        assert_equal(np.quantile(x, 0.5), 1.75)

    def test_correct_quantile_value(self):
        a = np.array([True])
        tf_quant = np.quantile(True, False)
        assert_equal(tf_quant, a[0])
        assert_equal(type(tf_quant), a.dtype)
        a = np.array([False, True, True])
        quant_res = np.quantile(a, a)
        assert_array_equal(quant_res, a)
        assert_equal(quant_res.dtype, a.dtype)

    def test_fraction(self):
        x = [Fraction(i, 2) for i in range(8)]
        q = np.quantile(x, 0)
        assert_equal(q, 0)
        assert_equal(type(q), Fraction)
        q = np.quantile(x, 1)
        assert_equal(q, Fraction(7, 2))
        assert_equal(type(q), Fraction)
        q = np.quantile(x, 0.5)
        assert_equal(q, 1.75)
        assert_equal(type(q), np.float64)
        q = np.quantile(x, Fraction(1, 2))
        assert_equal(q, Fraction(7, 4))
        assert_equal(type(q), Fraction)
        q = np.quantile(x, [Fraction(1, 2)])
        assert_equal(q, np.array([Fraction(7, 4)]))
        assert_equal(type(q), np.ndarray)
        q = np.quantile(x, [[Fraction(1, 2)]])
        assert_equal(q, np.array([[Fraction(7, 4)]]))
        assert_equal(type(q), np.ndarray)
        x = np.arange(8)
        assert_equal(np.quantile(x, Fraction(1, 2)), Fraction(7, 2))

    def test_complex(self):
        arr_c = np.array([0.5 + 3j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='G')
        assert_raises(TypeError, np.quantile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='D')
        assert_raises(TypeError, np.quantile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='F')
        assert_raises(TypeError, np.quantile, arr_c, 0.5)

    def test_no_p_overwrite(self):
        p0 = np.array([0, 0.75, 0.25, 0.5, 1.0])
        p = p0.copy()
        np.quantile(np.arange(100.0), p, method='midpoint')
        assert_array_equal(p, p0)
        p0 = p0.tolist()
        p = p.tolist()
        np.quantile(np.arange(100.0), p, method='midpoint')
        assert_array_equal(p, p0)

    @pytest.mark.parametrize('dtype', np.typecodes['AllInteger'])
    def test_quantile_preserve_int_type(self, dtype):
        res = np.quantile(np.array([1, 2], dtype=dtype), [0.5], method='nearest')
        assert res.dtype == dtype

    @pytest.mark.parametrize('method', quantile_methods)
    def test_quantile_monotonic(self, method):
        p0 = np.linspace(0, 1, 101)
        quantile = np.quantile(np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 1, 1, 9, 9, 9, 8, 8, 7]) * 0.1, p0, method=method)
        assert_equal(np.sort(quantile), quantile)
        quantile = np.quantile([0.0, 1.0, 2.0, 3.0], p0, method=method)
        assert_equal(np.sort(quantile), quantile)

    @hypothesis.given(arr=arrays(dtype=np.float64, shape=st.integers(min_value=3, max_value=1000), elements=st.floats(allow_infinity=False, allow_nan=False, min_value=-1e+300, max_value=1e+300)))
    def test_quantile_monotonic_hypo(self, arr):
        p0 = np.arange(0, 1, 0.01)
        quantile = np.quantile(arr, p0)
        assert_equal(np.sort(quantile), quantile)

    def test_quantile_scalar_nan(self):
        a = np.array([[10.0, 7.0, 4.0], [3.0, 2.0, 1.0]])
        a[0][1] = np.nan
        actual = np.quantile(a, 0.5)
        assert np.isscalar(actual)
        assert_equal(np.quantile(a, 0.5), np.nan)

    @pytest.mark.parametrize('method', quantile_methods)
    @pytest.mark.parametrize('alpha', [0.2, 0.5, 0.9])
    def test_quantile_identification_equation(self, method, alpha):
        rng = np.random.default_rng(4321)
        n = 102
        y = rng.random(n)
        x = np.quantile(y, alpha, method=method)
        if method in ('higher',):
            assert np.abs(np.mean(self.V(x, y, alpha))) > 0.1 / n
        elif int(n * alpha) == n * alpha:
            assert_allclose(np.mean(self.V(x, y, alpha)), 0, atol=1e-14)
        else:
            assert_allclose(np.mean(self.V(x, y, alpha)), 0, atol=1 / n / np.amin([alpha, 1 - alpha]))

    @pytest.mark.parametrize('method', quantile_methods)
    @pytest.mark.parametrize('alpha', [0.2, 0.5, 0.9])
    def test_quantile_add_and_multiply_constant(self, method, alpha):
        rng = np.random.default_rng(4321)
        n = 102
        y = rng.random(n)
        q = np.quantile(y, alpha, method=method)
        c = 13.5
        assert_allclose(np.quantile(c + y, alpha, method=method), c + q)
        assert_allclose(np.quantile(c * y, alpha, method=method), c * q)
        q = -np.quantile(-y, 1 - alpha, method=method)
        if method == 'inverted_cdf':
            if n * alpha == int(n * alpha) or np.round(n * alpha) == int(n * alpha) + 1:
                assert_allclose(q, np.quantile(y, alpha, method='higher'))
            else:
                assert_allclose(q, np.quantile(y, alpha, method='lower'))
        elif method == 'closest_observation':
            if n * alpha == int(n * alpha):
                assert_allclose(q, np.quantile(y, alpha, method='higher'))
            elif np.round(n * alpha) == int(n * alpha) + 1:
                assert_allclose(q, np.quantile(y, alpha + 1 / n, method='higher'))
            else:
                assert_allclose(q, np.quantile(y, alpha, method='lower'))
        elif method == 'interpolated_inverted_cdf':
            assert_allclose(q, np.quantile(y, alpha + 1 / n, method=method))
        elif method == 'nearest':
            if n * alpha == int(n * alpha):
                assert_allclose(q, np.quantile(y, alpha + 1 / n, method=method))
            else:
                assert_allclose(q, np.quantile(y, alpha, method=method))
        elif method == 'lower':
            assert_allclose(q, np.quantile(y, alpha, method='higher'))
        elif method == 'higher':
            assert_allclose(q, np.quantile(y, alpha, method='lower'))
        else:
            assert_allclose(q, np.quantile(y, alpha, method=method))