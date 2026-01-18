import pytest
import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyp_num
from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
from scipy.integrate._quadrature import _cumulative_simpson_unequal_intervals
from scipy.integrate._tanhsinh import _tanhsinh, _pair_cache
from scipy import stats, special as sc
from scipy.optimize._zeros_py import (_ECONVERGED, _ESIGNERR, _ECONVERR,  # noqa: F401
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class TestQuadrature:

    def quad(self, x, a, b, args):
        raise NotImplementedError

    def test_quadrature(self):

        def myfunc(x, n, z):
            return cos(n * x - z * sin(x)) / pi
        val, err = quadrature(myfunc, 0, pi, (2, 1.8))
        table_val = 0.30614353532540295
        assert_almost_equal(val, table_val, decimal=7)

    def test_quadrature_rtol(self):

        def myfunc(x, n, z):
            return 1e+90 * cos(n * x - z * sin(x)) / pi
        val, err = quadrature(myfunc, 0, pi, (2, 1.8), rtol=1e-10)
        table_val = 1e+90 * 0.30614353532540295
        assert_allclose(val, table_val, rtol=1e-10)

    def test_quadrature_miniter(self):

        def myfunc(x, n, z):
            return cos(n * x - z * sin(x)) / pi
        table_val = 0.30614353532540295
        for miniter in [5, 52]:
            val, err = quadrature(myfunc, 0, pi, (2, 1.8), miniter=miniter)
            assert_almost_equal(val, table_val, decimal=7)
            assert_(err < 1.0)

    def test_quadrature_single_args(self):

        def myfunc(x, n):
            return 1e+90 * cos(n * x - 1.8 * sin(x)) / pi
        val, err = quadrature(myfunc, 0, pi, args=2, rtol=1e-10)
        table_val = 1e+90 * 0.30614353532540295
        assert_allclose(val, table_val, rtol=1e-10)

    def test_romberg(self):

        def myfunc(x, n, z):
            return cos(n * x - z * sin(x)) / pi
        val = romberg(myfunc, 0, pi, args=(2, 1.8))
        table_val = 0.30614353532540295
        assert_almost_equal(val, table_val, decimal=7)

    def test_romberg_rtol(self):

        def myfunc(x, n, z):
            return 1e+19 * cos(n * x - z * sin(x)) / pi
        val = romberg(myfunc, 0, pi, args=(2, 1.8), rtol=1e-10)
        table_val = 1e+19 * 0.30614353532540295
        assert_allclose(val, table_val, rtol=1e-10)

    def test_romb(self):
        assert_equal(romb(np.arange(17)), 128)

    def test_romb_gh_3731(self):
        x = np.arange(2 ** 4 + 1)
        y = np.cos(0.2 * x)
        val = romb(y)
        val2, err = quad(lambda x: np.cos(0.2 * x), x.min(), x.max())
        assert_allclose(val, val2, rtol=1e-08, atol=0)
        with suppress_warnings() as sup:
            sup.filter(AccuracyWarning, 'divmax .4. exceeded')
            val3 = romberg(lambda x: np.cos(0.2 * x), x.min(), x.max(), divmax=4)
        assert_allclose(val, val3, rtol=1e-12, atol=0)

    def test_non_dtype(self):
        import math
        valmath = romberg(math.sin, 0, 1)
        expected_val = 0.45969769413185085
        assert_almost_equal(valmath, expected_val, decimal=7)

    def test_newton_cotes(self):
        """Test the first few degrees, for evenly spaced points."""
        n = 1
        wts, errcoff = newton_cotes(n, 1)
        assert_equal(wts, n * np.array([0.5, 0.5]))
        assert_almost_equal(errcoff, -n ** 3 / 12.0)
        n = 2
        wts, errcoff = newton_cotes(n, 1)
        assert_almost_equal(wts, n * np.array([1.0, 4.0, 1.0]) / 6.0)
        assert_almost_equal(errcoff, -n ** 5 / 2880.0)
        n = 3
        wts, errcoff = newton_cotes(n, 1)
        assert_almost_equal(wts, n * np.array([1.0, 3.0, 3.0, 1.0]) / 8.0)
        assert_almost_equal(errcoff, -n ** 5 / 6480.0)
        n = 4
        wts, errcoff = newton_cotes(n, 1)
        assert_almost_equal(wts, n * np.array([7.0, 32.0, 12.0, 32.0, 7.0]) / 90.0)
        assert_almost_equal(errcoff, -n ** 7 / 1935360.0)

    def test_newton_cotes2(self):
        """Test newton_cotes with points that are not evenly spaced."""
        x = np.array([0.0, 1.5, 2.0])
        y = x ** 2
        wts, errcoff = newton_cotes(x)
        exact_integral = 8.0 / 3
        numeric_integral = np.dot(wts, y)
        assert_almost_equal(numeric_integral, exact_integral)
        x = np.array([0.0, 1.4, 2.1, 3.0])
        y = x ** 2
        wts, errcoff = newton_cotes(x)
        exact_integral = 9.0
        numeric_integral = np.dot(wts, y)
        assert_almost_equal(numeric_integral, exact_integral)

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_simpson(self):
        y = np.arange(17)
        assert_equal(simpson(y), 128)
        assert_equal(simpson(y, dx=0.5), 64)
        assert_equal(simpson(y, x=np.linspace(0, 4, 17)), 32)
        y = np.arange(4)
        x = 2 ** y
        assert_equal(simpson(y, x=x, even='avg'), 13.875)
        assert_equal(simpson(y, x=x, even='first'), 13.75)
        assert_equal(simpson(y, x=x, even='last'), 14)
        x = np.linspace(1, 4, 4)

        def f(x):
            return x ** 2
        assert_allclose(simpson(f(x), x=x, even='simpson'), 21.0)
        assert_allclose(simpson(f(x), x=x, even='avg'), 21 + 1 / 6)
        x = np.linspace(1, 7, 4)
        assert_allclose(simpson(f(x), dx=2.0, even='simpson'), 114)
        assert_allclose(simpson(f(x), dx=2.0, even='avg'), 115 + 1 / 3)
        a = np.arange(16).reshape(4, 4)
        x = np.arange(64.0).reshape(4, 4, 4)
        y = f(x)
        for i in range(3):
            r = simpson(y, x=x, even='simpson', axis=i)
            it = np.nditer(a, flags=['multi_index'])
            for _ in it:
                idx = list(it.multi_index)
                idx.insert(i, slice(None))
                integral = x[tuple(idx)][-1] ** 3 / 3 - x[tuple(idx)][0] ** 3 / 3
                assert_allclose(r[it.multi_index], integral)
        x = np.arange(16).reshape(8, 2)
        y = f(x)
        for even in ['simpson', 'avg', 'first', 'last']:
            r = simpson(y, x=x, even=even, axis=-1)
            integral = 0.5 * (y[:, 1] + y[:, 0]) * (x[:, 1] - x[:, 0])
            assert_allclose(r, integral)
        a = np.arange(25).reshape(5, 5)
        x = np.arange(125).reshape(5, 5, 5)
        y = f(x)
        for i in range(3):
            r = simpson(y, x=x, axis=i)
            it = np.nditer(a, flags=['multi_index'])
            for _ in it:
                idx = list(it.multi_index)
                idx.insert(i, slice(None))
                integral = x[tuple(idx)][-1] ** 3 / 3 - x[tuple(idx)][0] ** 3 / 3
                assert_allclose(r[it.multi_index], integral)
        x = np.array([3])
        y = np.power(x, 2)
        assert_allclose(simpson(y, x=x, axis=0), 0.0)
        assert_allclose(simpson(y, x=x, axis=-1), 0.0)
        x = np.array([3, 3, 3, 3])
        y = np.power(x, 2)
        assert_allclose(simpson(y, x=x, axis=0), 0.0)
        assert_allclose(simpson(y, x=x, axis=-1), 0.0)
        x = np.array([[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]])
        y = np.power(x, 2)
        zero_axis = [0.0, 0.0, 0.0, 0.0]
        default_axis = [170 + 1 / 3] * 3
        assert_allclose(simpson(y, x=x, axis=0), zero_axis)
        assert_allclose(simpson(y, x=x, axis=-1), default_axis)
        x = np.array([[1, 2, 4, 8], [1, 2, 4, 8], [1, 8, 16, 32]])
        y = np.power(x, 2)
        zero_axis = [0.0, 136.0, 1088.0, 8704.0]
        default_axis = [170 + 1 / 3, 170 + 1 / 3, 32 ** 3 / 3 - 1 / 3]
        assert_allclose(simpson(y, x=x, axis=0), zero_axis)
        assert_allclose(simpson(y, x=x, axis=-1), default_axis)

    def test_simpson_deprecations(self):
        x = np.linspace(0, 3, 4)
        y = x ** 2
        with pytest.deprecated_call(match="The 'even' keyword is deprecated"):
            simpson(y, x=x, even='first')
        with pytest.deprecated_call(match='use keyword arguments'):
            simpson(y, x)

    @pytest.mark.parametrize('droplast', [False, True])
    def test_simpson_2d_integer_no_x(self, droplast):
        y = np.array([[2, 2, 4, 4, 8, 8, -4, 5], [4, 4, 2, -4, 10, 22, -2, 10]])
        if droplast:
            y = y[:, :-1]
        result = simpson(y, axis=-1)
        expected = simpson(np.array(y, dtype=np.float64), axis=-1)
        assert_equal(result, expected)

    def test_simps(self):
        y = np.arange(5)
        x = 2 ** y
        with pytest.deprecated_call(match='simpson'):
            assert_allclose(simpson(y, x=x, dx=0.5), simps(y, x=x, dx=0.5))