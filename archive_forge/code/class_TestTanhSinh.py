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
class TestTanhSinh:

    def f1(self, t):
        return t * np.log(1 + t)
    f1.ref = 0.25
    f1.b = 1

    def f2(self, t):
        return t ** 2 * np.arctan(t)
    f2.ref = (np.pi - 2 + 2 * np.log(2)) / 12
    f2.b = 1

    def f3(self, t):
        return np.exp(t) * np.cos(t)
    f3.ref = (np.exp(np.pi / 2) - 1) / 2
    f3.b = np.pi / 2

    def f4(self, t):
        a = np.sqrt(2 + t ** 2)
        return np.arctan(a) / ((1 + t ** 2) * a)
    f4.ref = 5 * np.pi ** 2 / 96
    f4.b = 1

    def f5(self, t):
        return np.sqrt(t) * np.log(t)
    f5.ref = -4 / 9
    f5.b = 1

    def f6(self, t):
        return np.sqrt(1 - t ** 2)
    f6.ref = np.pi / 4
    f6.b = 1

    def f7(self, t):
        return np.sqrt(t) / np.sqrt(1 - t ** 2)
    f7.ref = 2 * np.sqrt(np.pi) * sc.gamma(3 / 4) / sc.gamma(1 / 4)
    f7.b = 1

    def f8(self, t):
        return np.log(t) ** 2
    f8.ref = 2
    f8.b = 1

    def f9(self, t):
        return np.log(np.cos(t))
    f9.ref = -np.pi * np.log(2) / 2
    f9.b = np.pi / 2

    def f10(self, t):
        return np.sqrt(np.tan(t))
    f10.ref = np.pi * np.sqrt(2) / 2
    f10.b = np.pi / 2

    def f11(self, t):
        return 1 / (1 + t ** 2)
    f11.ref = np.pi / 2
    f11.b = np.inf

    def f12(self, t):
        return np.exp(-t) / np.sqrt(t)
    f12.ref = np.sqrt(np.pi)
    f12.b = np.inf

    def f13(self, t):
        return np.exp(-t ** 2 / 2)
    f13.ref = np.sqrt(np.pi / 2)
    f13.b = np.inf

    def f14(self, t):
        return np.exp(-t) * np.cos(t)
    f14.ref = 0.5
    f14.b = np.inf

    def f15(self, t):
        return np.sin(t) / t
    f15.ref = np.pi / 2
    f15.b = np.inf

    def error(self, res, ref, log=False):
        err = abs(res - ref)
        if not log:
            return err
        with np.errstate(divide='ignore'):
            return np.log10(err)

    def test_input_validation(self):
        f = self.f1
        message = '`f` must be callable.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(42, 0, f.b)
        message = '...must be True or False.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, log=2)
        message = '...must be real numbers.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 1 + 1j, f.b)
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, atol='ekki')
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, rtol=pytest)
        message = '...must be non-negative and finite.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, rtol=-1)
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, atol=np.inf)
        message = '...may not be positive infinity.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, rtol=np.inf, log=True)
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, atol=np.inf, log=True)
        message = '...must be integers.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, maxlevel=object())
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, maxfun=1 + 1j)
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, minlevel='migratory coconut')
        message = '...must be non-negative.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, maxlevel=-1)
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, maxfun=-1)
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, minlevel=-1)
        message = '...must be callable.'
        with pytest.raises(ValueError, match=message):
            _tanhsinh(f, 0, f.b, callback='elderberry')

    @pytest.mark.parametrize('limits, ref', [[(0, np.inf), 0.5], [(-np.inf, 0), 0.5], [(-np.inf, np.inf), 1], [(np.inf, -np.inf), -1], [(1, -1), stats.norm.cdf(-1) - stats.norm.cdf(1)]])
    def test_integral_transforms(self, limits, ref):
        dist = stats.norm()
        res = _tanhsinh(dist.pdf, *limits)
        assert_allclose(res.integral, ref)
        logres = _tanhsinh(dist.logpdf, *limits, log=True)
        assert_allclose(np.exp(logres.integral), ref)
        assert np.issubdtype(logres.integral.dtype, np.floating) if ref > 0 else np.issubdtype(logres.integral.dtype, np.complexfloating)
        assert_allclose(np.exp(logres.error), res.error, atol=1e-16)

    @pytest.mark.parametrize('f_number', range(1, 15))
    def test_basic(self, f_number):
        f = getattr(self, f'f{f_number}')
        rtol = 2e-08
        res = _tanhsinh(f, 0, f.b, rtol=rtol)
        assert_allclose(res.integral, f.ref, rtol=rtol)
        if f_number not in {14}:
            true_error = abs(self.error(res.integral, f.ref) / res.integral)
            assert true_error < res.error
        if f_number in {7, 10, 12}:
            return
        assert res.success
        assert res.status == 0

    @pytest.mark.parametrize('ref', (0.5, [0.4, 0.6]))
    @pytest.mark.parametrize('case', stats._distr_params.distcont)
    def test_accuracy(self, ref, case):
        distname, params = case
        if distname in {'dgamma', 'dweibull', 'laplace', 'kstwo'}:
            pytest.skip('tanh-sinh is not great for non-smooth integrands')
        dist = getattr(stats, distname)(*params)
        x = dist.interval(ref)
        res = _tanhsinh(dist.pdf, *x)
        assert_allclose(res.integral, ref)

    @pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])
    def test_vectorization(self, shape):
        rng = np.random.default_rng(82456839535679456794)
        a = rng.random(shape)
        b = rng.random(shape)
        p = rng.random(shape)
        n = np.prod(shape)

        def f(x, p):
            f.ncall += 1
            f.feval += 1 if x.size == n or x.ndim <= 1 else x.shape[-1]
            return x ** p
        f.ncall = 0
        f.feval = 0

        @np.vectorize
        def _tanhsinh_single(a, b, p):
            return _tanhsinh(lambda x: x ** p, a, b)
        res = _tanhsinh(f, a, b, args=(p,))
        refs = _tanhsinh_single(a, b, p).ravel()
        attrs = ['integral', 'error', 'success', 'status', 'nfev', 'maxlevel']
        for attr in attrs:
            ref_attr = [getattr(ref, attr) for ref in refs]
            res_attr = getattr(res, attr)
            assert_allclose(res_attr.ravel(), ref_attr, rtol=1e-15)
            assert_equal(res_attr.shape, shape)
        assert np.issubdtype(res.success.dtype, np.bool_)
        assert np.issubdtype(res.status.dtype, np.integer)
        assert np.issubdtype(res.nfev.dtype, np.integer)
        assert np.issubdtype(res.maxlevel.dtype, np.integer)
        assert_equal(np.max(res.nfev), f.feval)
        assert np.max(res.maxlevel) >= 2
        assert_equal(np.max(res.maxlevel), f.ncall)

    def test_flags(self):

        def f(xs, js):
            f.nit += 1
            funcs = [lambda x: np.exp(-x ** 2), lambda x: np.exp(x), lambda x: np.full_like(x, np.nan)[()]]
            res = [funcs[j](x) for x, j in zip(xs, js.ravel())]
            return res
        f.nit = 0
        args = (np.arange(3, dtype=np.int64),)
        res = _tanhsinh(f, [np.inf] * 3, [-np.inf] * 3, maxlevel=5, args=args)
        ref_flags = np.array([0, -2, -3])
        assert_equal(res.status, ref_flags)

    def test_convergence(self):
        f = self.f1
        last_logerr = 0
        for i in range(4):
            res = _tanhsinh(f, 0, f.b, minlevel=0, maxlevel=i)
            logerr = self.error(res.integral, f.ref, log=True)
            assert logerr < last_logerr * 2 or logerr < -15.5
            last_logerr = logerr

    def test_options_and_result_attributes(self):

        def f(x):
            f.calls += 1
            f.feval += np.size(x)
            return self.f2(x)
        f.ref = self.f2.ref
        f.b = self.f2.b
        default_rtol = 1e-12
        default_atol = f.ref * default_rtol
        f.feval, f.calls = (0, 0)
        ref = _tanhsinh(f, 0, f.b)
        assert self.error(ref.integral, f.ref) < ref.error < default_atol
        assert ref.nfev == f.feval
        ref.calls = f.calls
        assert ref.success
        assert ref.status == 0
        f.feval, f.calls = (0, 0)
        maxlevel = ref.maxlevel
        res = _tanhsinh(f, 0, f.b, maxlevel=maxlevel)
        res.calls = f.calls
        assert res == ref
        f.feval, f.calls = (0, 0)
        maxlevel -= 1
        assert maxlevel >= 2
        res = _tanhsinh(f, 0, f.b, maxlevel=maxlevel)
        assert self.error(res.integral, f.ref) < res.error > default_atol
        assert res.nfev == f.feval < ref.nfev
        assert f.calls == ref.calls - 1
        assert not res.success
        assert res.status == _ECONVERR
        ref = res
        ref.calls = f.calls
        f.feval, f.calls = (0, 0)
        atol = np.nextafter(ref.error, np.inf)
        res = _tanhsinh(f, 0, f.b, rtol=0, atol=atol)
        assert res.integral == ref.integral
        assert res.error == ref.error
        assert res.nfev == f.feval == ref.nfev
        assert f.calls == ref.calls
        assert res.success
        assert res.status == 0
        f.feval, f.calls = (0, 0)
        atol = np.nextafter(ref.error, -np.inf)
        res = _tanhsinh(f, 0, f.b, rtol=0, atol=atol)
        assert self.error(res.integral, f.ref) < res.error < atol
        assert res.nfev == f.feval > ref.nfev
        assert f.calls > ref.calls
        assert res.success
        assert res.status == 0
        f.feval, f.calls = (0, 0)
        rtol = np.nextafter(ref.error / ref.integral, np.inf)
        res = _tanhsinh(f, 0, f.b, rtol=rtol)
        assert res.integral == ref.integral
        assert res.error == ref.error
        assert res.nfev == f.feval == ref.nfev
        assert f.calls == ref.calls
        assert res.success
        assert res.status == 0
        f.feval, f.calls = (0, 0)
        rtol = np.nextafter(ref.error / ref.integral, -np.inf)
        res = _tanhsinh(f, 0, f.b, rtol=rtol)
        assert self.error(res.integral, f.ref) / f.ref < res.error / res.integral < rtol
        assert res.nfev == f.feval > ref.nfev
        assert f.calls > ref.calls
        assert res.success
        assert res.status == 0

    @pytest.mark.parametrize('rtol', [0.0001, 1e-14])
    def test_log(self, rtol):
        dist = stats.norm()
        test_tols = dict(atol=1e-18, rtol=1e-15)
        res = _tanhsinh(dist.logpdf, -1, 2, log=True, rtol=np.log(rtol))
        ref = _tanhsinh(dist.pdf, -1, 2, rtol=rtol)
        assert_allclose(np.exp(res.integral), ref.integral, **test_tols)
        assert_allclose(np.exp(res.error), ref.error, **test_tols)
        assert res.nfev == ref.nfev

        def f(x):
            return -dist.logpdf(x) * dist.pdf(x)

        def logf(x):
            return np.log(dist.logpdf(x) + 0j) + dist.logpdf(x) + np.pi * 1j
        res = _tanhsinh(logf, -np.inf, np.inf, log=True)
        ref = _tanhsinh(f, -np.inf, np.inf)
        with np.errstate(all='ignore'):
            assert_allclose(np.exp(res.integral), ref.integral, **test_tols)
            assert_allclose(np.exp(res.error), ref.error, **test_tols)
        assert res.nfev == ref.nfev

    def test_complex(self):

        def f(x):
            return np.exp(1j * x)
        res = _tanhsinh(f, 0, np.pi / 4)
        ref = np.sqrt(2) / 2 + (1 - np.sqrt(2) / 2) * 1j
        assert_allclose(res.integral, ref)
        dist1 = stats.norm(scale=1)
        dist2 = stats.norm(scale=2)

        def f(x):
            return dist1.pdf(x) + 1j * dist2.pdf(x)
        res = _tanhsinh(f, np.inf, -np.inf)
        assert_allclose(res.integral, -(1 + 1j))

    @pytest.mark.parametrize('maxlevel', range(4))
    def test_minlevel(self, maxlevel):

        def f(x):
            f.calls += 1
            f.feval += np.size(x)
            f.x = np.concatenate((f.x, x.ravel()))
            return self.f2(x)
        f.feval, f.calls, f.x = (0, 0, np.array([]))
        ref = _tanhsinh(f, 0, self.f2.b, minlevel=0, maxlevel=maxlevel)
        ref_x = np.sort(f.x)
        for minlevel in range(0, maxlevel + 1):
            f.feval, f.calls, f.x = (0, 0, np.array([]))
            options = dict(minlevel=minlevel, maxlevel=maxlevel)
            res = _tanhsinh(f, 0, self.f2.b, **options)
            assert_allclose(res.integral, ref.integral, rtol=4e-16)
            assert_allclose(res.error, ref.error, atol=4e-16 * ref.integral)
            assert res.nfev == f.feval == len(f.x)
            assert f.calls == maxlevel - minlevel + 1 + 1
            assert res.status == ref.status
            assert_equal(ref_x, np.sort(f.x))

    def test_improper_integrals(self):

        def f(x):
            return np.exp(-x ** 2)
        a = [-np.inf, 0, -np.inf, np.inf, -20, -np.inf, -20]
        b = [np.inf, np.inf, 0, -np.inf, 20, 20, np.inf]
        ref = np.sqrt(np.pi)
        res = _tanhsinh(f, a, b)
        assert_allclose(res.integral, [ref, ref / 2, ref / 2, -ref, ref, ref, ref])

    @pytest.mark.parametrize('limits', ((0, 3), ([-np.inf, 0], [3, 3])))
    @pytest.mark.parametrize('dtype', (np.float32, np.float64))
    def test_dtype(self, limits, dtype):
        a, b = np.asarray(limits, dtype=dtype)[()]

        def f(x):
            assert x.dtype == dtype
            return np.exp(x)
        rtol = 1e-12 if dtype == np.float64 else 1e-05
        res = _tanhsinh(f, a, b, rtol=rtol)
        assert res.integral.dtype == dtype
        assert res.error.dtype == dtype
        assert np.all(res.success)
        assert_allclose(res.integral, np.exp(b) - np.exp(a), rtol=rtol)

    def test_maxiter_callback(self):
        a, b = (-np.inf, np.inf)

        def f(x):
            return np.exp(-x * x)
        minlevel, maxlevel = (0, 2)
        maxiter = maxlevel - minlevel + 1
        kwargs = dict(minlevel=minlevel, maxlevel=maxlevel, rtol=1e-15)
        res = _tanhsinh(f, a, b, **kwargs)
        assert not res.success
        assert res.maxlevel == maxlevel

        def callback(res):
            callback.iter += 1
            callback.res = res
            assert hasattr(res, 'integral')
            assert res.status == 1
            if callback.iter == maxiter:
                raise StopIteration
        callback.iter = -1
        callback.res = None
        del kwargs['maxlevel']
        res2 = _tanhsinh(f, a, b, **kwargs, callback=callback)
        for key in res.keys():
            if key == 'status':
                assert callback.res[key] == 1
                assert res[key] == -2
                assert res2[key] == -4
            else:
                assert res2[key] == callback.res[key] == res[key]

    def test_jumpstart(self):
        a, b = (-np.inf, np.inf)

        def f(x):
            return np.exp(-x * x)

        def callback(res):
            callback.integrals.append(res.integral)
            callback.errors.append(res.error)
        callback.integrals = []
        callback.errors = []
        maxlevel = 4
        _tanhsinh(f, a, b, minlevel=0, maxlevel=maxlevel, callback=callback)
        integrals = []
        errors = []
        for i in range(maxlevel + 1):
            res = _tanhsinh(f, a, b, minlevel=i, maxlevel=i)
            integrals.append(res.integral)
            errors.append(res.error)
        assert_allclose(callback.integrals[1:], integrals, rtol=1e-15)
        assert_allclose(callback.errors[1:], errors, rtol=1e-15, atol=1e-16)

    def test_special_cases(self):

        def f(x):
            assert np.issubdtype(x.dtype, np.floating)
            return x ** 99
        res = _tanhsinh(f, 0, 1)
        assert res.success
        assert_allclose(res.integral, 1 / 100)
        res = _tanhsinh(f, 0, 1, maxlevel=0)
        assert res.integral > 0
        assert_equal(res.error, np.nan)
        res = _tanhsinh(f, 0, 1, maxlevel=1)
        assert res.integral > 0
        assert_equal(res.error, np.nan)
        res = _tanhsinh(f, 1, 1)
        assert res.success
        assert res.maxlevel == -1
        assert_allclose(res.integral, 0)

        def f(x, c):
            return x ** c
        res = _tanhsinh(f, 0, 1, args=99)
        assert_allclose(res.integral, 1 / 100)
        a = [np.nan, 0, 0, 0]
        b = [1, np.nan, 1, 1]
        c = [1, 1, np.nan, 1]
        res = _tanhsinh(f, a, b, args=(c,))
        assert_allclose(res.integral, [np.nan, np.nan, np.nan, 0.5])
        assert_allclose(res.error[:3], np.nan)
        assert_equal(res.status, [-3, -3, -3, 0])
        assert_equal(res.success, [False, False, False, True])
        assert_equal(res.nfev[:3], 1)
        _pair_cache.xjc = np.empty(0)
        _pair_cache.wj = np.empty(0)
        _pair_cache.indices = [0]
        _pair_cache.h0 = None
        res = _tanhsinh(lambda x: x * 1j, 0, 1)
        assert_allclose(res.integral, 0.5 * 1j)
        res = _tanhsinh(lambda x: x, 0, 1)
        assert_allclose(res.integral, 0.5)
        shape = (0, 3)
        res = _tanhsinh(lambda x: x, 0, np.zeros(shape))
        attrs = ['integral', 'error', 'success', 'status', 'nfev', 'maxlevel']
        for attr in attrs:
            assert_equal(res[attr].shape, shape)