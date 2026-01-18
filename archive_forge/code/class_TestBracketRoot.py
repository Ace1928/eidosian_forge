import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
class TestBracketRoot:

    @pytest.mark.parametrize('seed', (615655101, 3141866013, 238075752))
    @pytest.mark.parametrize('use_min', (False, True))
    @pytest.mark.parametrize('other_side', (False, True))
    @pytest.mark.parametrize('fix_one_side', (False, True))
    def test_nfev_expected(self, seed, use_min, other_side, fix_one_side):
        rng = np.random.default_rng(seed)
        a, d, factor = rng.random(size=3) * [100000.0, 10, 5]
        factor = 1 + factor
        b = a + d

        def f(x):
            f.count += 1
            return x
        if use_min:
            min = -rng.random()
            n = np.ceil(np.log(-(a - min) / min) / np.log(factor))
            l, u = (min + (a - min) * factor ** (-n), min + (a - min) * factor ** (-(n - 1)))
            kwargs = dict(a=a, b=b, factor=factor, min=min)
        else:
            n = np.ceil(np.log(b / d) / np.log(factor))
            l, u = (b - d * factor ** n, b - d * factor ** (n - 1))
            kwargs = dict(a=a, b=b, factor=factor)
        if other_side:
            kwargs['a'], kwargs['b'] = (-kwargs['b'], -kwargs['a'])
            l, u = (-u, -l)
            if 'min' in kwargs:
                kwargs['max'] = -kwargs.pop('min')
        if fix_one_side:
            if other_side:
                kwargs['min'] = -b
            else:
                kwargs['max'] = b
        f.count = 0
        res = zeros._bracket_root(f, **kwargs)
        if not fix_one_side:
            assert res.nfev == 2 * (res.nit + 1) == 2 * (f.count - 1) == 2 * (n + 1)
        else:
            assert res.nfev == res.nit + 1 + 1 == f.count - 1 + 1 == n + 1 + 1
        bracket = np.asarray([res.xl, res.xr])
        assert_allclose(bracket, (l, u))
        f_bracket = np.asarray([res.fl, res.fr])
        assert_allclose(f_bracket, f(bracket))
        assert res.xr > res.xl
        signs = np.sign(f_bracket)
        assert signs[0] == -signs[1]
        assert res.status == 0
        assert res.success

    def f(self, q, p):
        return stats.norm.cdf(q) - p

    @pytest.mark.parametrize('p', [0.6, np.linspace(0.05, 0.95, 10)])
    @pytest.mark.parametrize('min', [-5, None])
    @pytest.mark.parametrize('max', [5, None])
    @pytest.mark.parametrize('factor', [1.2, 2])
    def test_basic(self, p, min, max, factor):
        res = zeros._bracket_root(self.f, -0.01, 0.01, min=min, max=max, factor=factor, args=(p,))
        assert_equal(-np.sign(res.fl), np.sign(res.fr))

    @pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])
    def test_vectorization(self, shape):
        p = np.linspace(-0.05, 1.05, 12).reshape(shape) if shape else 0.6
        args = (p,)
        maxiter = 10

        @np.vectorize
        def bracket_root_single(a, b, min, max, factor, p):
            return zeros._bracket_root(self.f, a, b, min=min, max=max, factor=factor, args=(p,), maxiter=maxiter)

        def f(*args, **kwargs):
            f.f_evals += 1
            return self.f(*args, **kwargs)
        f.f_evals = 0
        rng = np.random.default_rng(2348234)
        a = -rng.random(size=shape)
        b = rng.random(size=shape)
        min, max = (1000.0 * a, 1000.0 * b)
        if shape:
            i = rng.random(size=shape) > 0.5
            min[i], max[i] = (-np.inf, np.inf)
        factor = rng.random(size=shape) + 1.5
        res = zeros._bracket_root(f, a, b, min=min, max=max, factor=factor, args=args, maxiter=maxiter)
        refs = bracket_root_single(a, b, min, max, factor, p).ravel()
        attrs = ['xl', 'xr', 'fl', 'fr', 'success', 'nfev', 'nit']
        for attr in attrs:
            ref_attr = [getattr(ref, attr) for ref in refs]
            res_attr = getattr(res, attr)
            assert_allclose(res_attr.ravel(), ref_attr)
            assert_equal(res_attr.shape, shape)
        assert np.issubdtype(res.success.dtype, np.bool_)
        if shape:
            assert np.all(res.success[1:-1])
        assert np.issubdtype(res.status.dtype, np.integer)
        assert np.issubdtype(res.nfev.dtype, np.integer)
        assert np.issubdtype(res.nit.dtype, np.integer)
        assert_equal(np.max(res.nit), f.f_evals - 2)
        assert_array_less(res.xl, res.xr)
        assert_allclose(res.fl, self.f(res.xl, *args))
        assert_allclose(res.fr, self.f(res.xr, *args))

    def test_flags(self):

        def f(xs, js):
            funcs = [lambda x: x - 1.5, lambda x: x - 1000, lambda x: x - 1000, lambda x: np.nan]
            return [funcs[j](x) for x, j in zip(xs, js)]
        args = (np.arange(4, dtype=np.int64),)
        res = zeros._bracket_root(f, a=[-1, -1, -1, -1], b=[1, 1, 1, 1], min=[-np.inf, -1, -np.inf, -np.inf], max=[np.inf, 1, np.inf, np.inf], args=args, maxiter=3)
        ref_flags = np.array([zeros._ECONVERGED, zeros._ELIMITS, zeros._ECONVERR, zeros._EVALUEERR])
        assert_equal(res.status, ref_flags)

    @pytest.mark.parametrize('root', (0.622, [0.622, 0.623]))
    @pytest.mark.parametrize('min', [-5, None])
    @pytest.mark.parametrize('max', [5, None])
    @pytest.mark.parametrize('dtype', (np.float16, np.float32, np.float64))
    def test_dtype(self, root, min, max, dtype):
        min = min if min is None else dtype(min)
        max = max if max is None else dtype(max)
        root = dtype(root)

        def f(x, root):
            return ((x - root) ** 3).astype(dtype)
        bracket = np.asarray([-0.01, 0.01], dtype=dtype)
        res = zeros._bracket_root(f, *bracket, min=min, max=max, args=(root,))
        assert np.all(res.success)
        assert res.xl.dtype == res.xr.dtype == dtype
        assert res.fl.dtype == res.fr.dtype == dtype

    def test_input_validation(self):
        message = '`func` must be callable.'
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(None, -4, 4)
        message = '...must be numeric and real.'
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4 + 1j, 4)
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 'hello')
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 4, min=zeros)
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 4, max=object())
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 4, factor=sum)
        message = 'All elements of `factor` must be greater than 1.'
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 4, factor=0.5)
        message = '`min <= a < b <= max` must be True'
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, 4, -4)
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 4, max=np.nan)
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, -4, 4, min=10)
        message = 'shape mismatch: objects cannot be broadcast'
        with pytest.raises(ValueError, match=message):
            zeros._bracket_root(lambda x: x, [-2, -3], [3, 4, 5])
        message = '`maxiter` must be a non-negative integer.'
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, maxiter=1.5)
        with pytest.raises(ValueError, match=message):
            zeros._chandrupatla(lambda x: x, -4, 4, maxiter=-1)

    def test_special_cases(self):

        def f(x):
            assert np.issubdtype(x.dtype, np.floating)
            return x ** 99 - 1
        res = zeros._bracket_root(f, -7, 5)
        assert res.success

        def f(x):
            return x - 10
        bracket = (-3, 5)
        res = zeros._bracket_root(f, *bracket, maxiter=0)
        assert res.xl, res.xr == bracket
        assert res.nit == 0
        assert res.nfev == 2
        assert res.status == -2

        def f(x, c):
            return c * x - 1
        res = zeros._bracket_root(f, -1, 1, args=3)
        assert res.success
        assert_allclose(res.fl, f(res.xl, 3))

        def f(x):
            f.count += 1
            return x
        f.count = 0
        zeros._bracket_root(f, -10, 20)
        assert_equal(f.count, 2)
        f.count = 0
        res = zeros._bracket_root(f, 5, 10, factor=2)
        bracket = (res.xl, res.xr)
        assert_equal(res.nfev, 4)
        assert_allclose(bracket, (0, 5), atol=1e-15)
        with np.errstate(over='ignore'):
            res = zeros._bracket_root(f, 5, 10, min=0)
        bracket = (res.xl, res.xr)
        assert_allclose(bracket[0], 0, atol=1e-15)
        with np.errstate(over='ignore'):
            res = zeros._bracket_root(f, -10, -5, max=0)
        bracket = (res.xl, res.xr)
        assert_allclose(bracket[1], 0, atol=1e-15)
        with np.errstate(over='ignore'):
            res = zeros._bracket_root(f, 5, 10, min=1)
        assert not res.success