import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
class TestChandrupatla(TestScalarRootFinders):

    def f(self, q, p):
        return stats.norm.cdf(q) - p

    @pytest.mark.parametrize('p', [0.6, np.linspace(-0.05, 1.05, 10)])
    def test_basic(self, p):
        res = zeros._chandrupatla(self.f, -5, 5, args=(p,))
        ref = stats.norm().ppf(p)
        np.testing.assert_allclose(res.x, ref)
        assert res.x.shape == ref.shape

    @pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])
    def test_vectorization(self, shape):
        p = np.linspace(-0.05, 1.05, 12).reshape(shape) if shape else 0.6
        args = (p,)

        @np.vectorize
        def chandrupatla_single(p):
            return zeros._chandrupatla(self.f, -5, 5, args=(p,))

        def f(*args, **kwargs):
            f.f_evals += 1
            return self.f(*args, **kwargs)
        f.f_evals = 0
        res = zeros._chandrupatla(f, -5, 5, args=args)
        refs = chandrupatla_single(p).ravel()
        ref_x = [ref.x for ref in refs]
        assert_allclose(res.x.ravel(), ref_x)
        assert_equal(res.x.shape, shape)
        ref_fun = [ref.fun for ref in refs]
        assert_allclose(res.fun.ravel(), ref_fun)
        assert_equal(res.fun.shape, shape)
        assert_equal(res.fun, self.f(res.x, *args))
        ref_success = [ref.success for ref in refs]
        assert_equal(res.success.ravel(), ref_success)
        assert_equal(res.success.shape, shape)
        assert np.issubdtype(res.success.dtype, np.bool_)
        ref_flag = [ref.status for ref in refs]
        assert_equal(res.status.ravel(), ref_flag)
        assert_equal(res.status.shape, shape)
        assert np.issubdtype(res.status.dtype, np.integer)
        ref_nfev = [ref.nfev for ref in refs]
        assert_equal(res.nfev.ravel(), ref_nfev)
        assert_equal(np.max(res.nfev), f.f_evals)
        assert_equal(res.nfev.shape, res.fun.shape)
        assert np.issubdtype(res.nfev.dtype, np.integer)
        ref_nit = [ref.nit for ref in refs]
        assert_equal(res.nit.ravel(), ref_nit)
        assert_equal(np.max(res.nit), f.f_evals - 2)
        assert_equal(res.nit.shape, res.fun.shape)
        assert np.issubdtype(res.nit.dtype, np.integer)
        ref_xl = [ref.xl for ref in refs]
        assert_allclose(res.xl.ravel(), ref_xl)
        assert_equal(res.xl.shape, shape)
        ref_xr = [ref.xr for ref in refs]
        assert_allclose(res.xr.ravel(), ref_xr)
        assert_equal(res.xr.shape, shape)
        assert_array_less(res.xl, res.xr)
        finite = np.isfinite(res.x)
        assert np.all((res.x[finite] == res.xl[finite]) | (res.x[finite] == res.xr[finite]))
        ref_fl = [ref.fl for ref in refs]
        assert_allclose(res.fl.ravel(), ref_fl)
        assert_equal(res.fl.shape, shape)
        assert_allclose(res.fl, self.f(res.xl, *args))
        ref_fr = [ref.fr for ref in refs]
        assert_allclose(res.fr.ravel(), ref_fr)
        assert_equal(res.fr.shape, shape)
        assert_allclose(res.fr, self.f(res.xr, *args))
        assert np.all(np.abs(res.fun[finite]) == np.minimum(np.abs(res.fl[finite]), np.abs(res.fr[finite])))

    def test_flags(self):

        def f(xs, js):
            funcs = [lambda x: x - 2.5, lambda x: x - 10, lambda x: (x - 0.1) ** 3, lambda x: np.nan]
            return [funcs[j](x) for x, j in zip(xs, js)]
        args = (np.arange(4, dtype=np.int64),)
        res = zeros._chandrupatla(f, [0] * 4, [np.pi] * 4, args=args, maxiter=2)
        ref_flags = np.array([zeros._ECONVERGED, zeros._ESIGNERR, zeros._ECONVERR, zeros._EVALUEERR])
        assert_equal(res.status, ref_flags)

    def test_convergence(self):
        rng = np.random.default_rng(2585255913088665241)
        p = rng.random(size=3)
        bracket = (-5, 5)
        args = (p,)
        kwargs0 = dict(args=args, xatol=0, xrtol=0, fatol=0, frtol=0)
        kwargs = kwargs0.copy()
        kwargs['xatol'] = 0.001
        res1 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(res1.xr - res1.xl, 0.001)
        kwargs['xatol'] = 1e-06
        res2 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(res2.xr - res2.xl, 1e-06)
        assert_array_less(res2.xr - res2.xl, res1.xr - res1.xl)
        kwargs = kwargs0.copy()
        kwargs['xrtol'] = 0.001
        res1 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(res1.xr - res1.xl, 0.001 * np.abs(res1.x))
        kwargs['xrtol'] = 1e-06
        res2 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(res2.xr - res2.xl, 1e-06 * np.abs(res2.x))
        assert_array_less(res2.xr - res2.xl, res1.xr - res1.xl)
        kwargs = kwargs0.copy()
        kwargs['fatol'] = 0.001
        res1 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(np.abs(res1.fun), 0.001)
        kwargs['fatol'] = 1e-06
        res2 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(np.abs(res2.fun), 1e-06)
        assert_array_less(np.abs(res2.fun), np.abs(res1.fun))
        kwargs = kwargs0.copy()
        kwargs['frtol'] = 0.001
        x1, x2 = bracket
        f0 = np.minimum(abs(self.f(x1, *args)), abs(self.f(x2, *args)))
        res1 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(np.abs(res1.fun), 0.001 * f0)
        kwargs['frtol'] = 1e-06
        res2 = zeros._chandrupatla(self.f, *bracket, **kwargs)
        assert_array_less(np.abs(res2.fun), 1e-06 * f0)
        assert_array_less(np.abs(res2.fun), np.abs(res1.fun))

    def test_maxiter_callback(self):
        p = 0.612814
        bracket = (-5, 5)
        maxiter = 5

        def f(q, p):
            res = stats.norm().cdf(q) - p
            f.x = q
            f.fun = res
            return res
        f.x = None
        f.fun = None
        res = zeros._chandrupatla(f, *bracket, args=(p,), maxiter=maxiter)
        assert not np.any(res.success)
        assert np.all(res.nfev == maxiter + 2)
        assert np.all(res.nit == maxiter)

        def callback(res):
            callback.iter += 1
            callback.res = res
            assert hasattr(res, 'x')
            if callback.iter == 0:
                assert (res.xl, res.xr) == bracket
            else:
                changed = (res.xl == callback.xl) & (res.xr != callback.xr) | (res.xl != callback.xl) & (res.xr == callback.xr)
                assert np.all(changed)
            callback.xl = res.xl
            callback.xr = res.xr
            assert res.status == zeros._EINPROGRESS
            assert_equal(self.f(res.xl, p), res.fl)
            assert_equal(self.f(res.xr, p), res.fr)
            assert_equal(self.f(res.x, p), res.fun)
            if callback.iter == maxiter:
                raise StopIteration
        callback.iter = -1
        callback.res = None
        callback.xl = None
        callback.xr = None
        res2 = zeros._chandrupatla(f, *bracket, args=(p,), callback=callback)
        for key in res.keys():
            if key == 'status':
                assert res[key] == zeros._ECONVERR
                assert callback.res[key] == zeros._EINPROGRESS
                assert res2[key] == zeros._ECALLBACK
            else:
                assert res2[key] == callback.res[key] == res[key]

    @pytest.mark.parametrize('case', optimize._tstutils._CHANDRUPATLA_TESTS)
    def test_nit_expected(self, case):
        f, bracket, root, nfeval, id = case
        res = zeros._chandrupatla(f, *bracket, xrtol=4e-10, xatol=1e-05)
        assert_allclose(res.fun, f(root), rtol=1e-08, atol=0.002)
        assert_equal(res.nfev, nfeval)

    @pytest.mark.parametrize('root', (0.622, [0.622, 0.623]))
    @pytest.mark.parametrize('dtype', (np.float16, np.float32, np.float64))
    def test_dtype(self, root, dtype):
        root = dtype(root)

        def f(x, root):
            return ((x - root) ** 3).astype(dtype)
        res = zeros._chandrupatla(f, dtype(-3), dtype(5), args=(root,), xatol=0.001)
        assert res.x.dtype == dtype
        assert np.allclose(res.x, root, atol=0.001) or np.all(res.fun == 0)

    def test_input_validation(self):
        message = '`func` must be callable.'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(None, -4, 4)
        message = 'Abscissae and function output must be real numbers.'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4 + 1j, 4)
        message = 'shape mismatch: objects cannot be broadcast'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, [-2, -3], [3, 4, 5])
        message = 'The shape of the array returned by `func`...'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: [x[0], x[1], x[1]], [-3, -3], [5, 5])
        message = 'Tolerances must be non-negative scalars.'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, xatol=-1)
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, xrtol=np.nan)
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, fatol='ekki')
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, frtol=np.nan)
        message = '`maxiter` must be a non-negative integer.'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, maxiter=1.5)
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, maxiter=-1)
        message = '`callback` must be callable.'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, callback='shrubbery')

    def test_special_cases(self):

        def f(x):
            assert np.issubdtype(x.dtype, np.floating)
            return x ** 99 - 1
        res = zeros._chandrupatla(f, -7, 5)
        assert res.success
        assert_allclose(res.x, 1)

        def f(x):
            return x ** 2 - 1
        res = zeros._chandrupatla(f, 1, 1)
        assert res.success
        assert_equal(res.x, 1)

        def f(x):
            return 1 / x
        with np.errstate(invalid='ignore'):
            res = zeros._chandrupatla(f, np.inf, np.inf)
        assert res.success
        assert_equal(res.x, np.inf)

        def f(x):
            return x ** 3 - 1
        bracket = (-3, 5)
        res = zeros._chandrupatla(f, *bracket, maxiter=0)
        assert res.xl, res.xr == bracket
        assert res.nit == 0
        assert res.nfev == 2
        assert res.status == -2
        assert res.x == -3
        res = zeros._chandrupatla(f, *bracket, maxiter=1)
        assert res.success
        assert res.status == 0
        assert res.nit == 1
        assert res.nfev == 3
        assert_allclose(res.x, 1)

        def f(x, c):
            return c * x - 1
        res = zeros._chandrupatla(f, -1, 1, args=3)
        assert_allclose(res.x, 1 / 3)