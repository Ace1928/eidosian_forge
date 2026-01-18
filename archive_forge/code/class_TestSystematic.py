import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools
from scipy._lib import _pep440
import scipy.special as sc
from scipy.special._testutils import (
from scipy.special._mptestutils import (
from scipy.special._ufuncs import (
@pytest.mark.slow
@check_version(mpmath, '0.17')
class TestSystematic:

    def test_airyai(self):
        assert_mpmath_equal(lambda z: sc.airy(z)[0], mpmath.airyai, [Arg(-100000000.0, 100000000.0)], rtol=1e-05)
        assert_mpmath_equal(lambda z: sc.airy(z)[0], mpmath.airyai, [Arg(-1000.0, 1000.0)])

    def test_airyai_complex(self):
        assert_mpmath_equal(lambda z: sc.airy(z)[0], mpmath.airyai, [ComplexArg()])

    def test_airyai_prime(self):
        assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z: mpmath.airyai(z, derivative=1), [Arg(-100000000.0, 100000000.0)], rtol=1e-05)
        assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z: mpmath.airyai(z, derivative=1), [Arg(-1000.0, 1000.0)])

    def test_airyai_prime_complex(self):
        assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z: mpmath.airyai(z, derivative=1), [ComplexArg()])

    def test_airybi(self):
        assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z: mpmath.airybi(z), [Arg(-100000000.0, 100000000.0)], rtol=1e-05)
        assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z: mpmath.airybi(z), [Arg(-1000.0, 1000.0)])

    def test_airybi_complex(self):
        assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z: mpmath.airybi(z), [ComplexArg()])

    def test_airybi_prime(self):
        assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z: mpmath.airybi(z, derivative=1), [Arg(-100000000.0, 100000000.0)], rtol=1e-05)
        assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z: mpmath.airybi(z, derivative=1), [Arg(-1000.0, 1000.0)])

    def test_airybi_prime_complex(self):
        assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z: mpmath.airybi(z, derivative=1), [ComplexArg()])

    def test_bei(self):
        assert_mpmath_equal(sc.bei, exception_to_nan(lambda z: mpmath.bei(0, z, **HYPERKW)), [Arg(-1000.0, 1000.0)])

    def test_ber(self):
        assert_mpmath_equal(sc.ber, exception_to_nan(lambda z: mpmath.ber(0, z, **HYPERKW)), [Arg(-1000.0, 1000.0)])

    def test_bernoulli(self):
        assert_mpmath_equal(lambda n: sc.bernoulli(int(n))[int(n)], lambda n: float(mpmath.bernoulli(int(n))), [IntArg(0, 13000)], rtol=1e-09, n=13000)

    def test_besseli(self):
        assert_mpmath_equal(sc.iv, exception_to_nan(lambda v, z: mpmath.besseli(v, z, **HYPERKW)), [Arg(-1e+100, 1e+100), Arg()], atol=1e-270)

    def test_besseli_complex(self):
        assert_mpmath_equal(lambda v, z: sc.iv(v.real, z), exception_to_nan(lambda v, z: mpmath.besseli(v, z, **HYPERKW)), [Arg(-1e+100, 1e+100), ComplexArg()])

    def test_besselj(self):
        assert_mpmath_equal(sc.jv, exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)), [Arg(-1e+100, 1e+100), Arg(-1000.0, 1000.0)], ignore_inf_sign=True)
        assert_mpmath_equal(sc.jv, exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)), [Arg(-1e+100, 1e+100), Arg(-100000000.0, 100000000.0)], ignore_inf_sign=True, rtol=1e-05)

    def test_besselj_complex(self):
        assert_mpmath_equal(lambda v, z: sc.jv(v.real, z), exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)), [Arg(), ComplexArg()])

    def test_besselk(self):
        assert_mpmath_equal(sc.kv, mpmath.besselk, [Arg(-200, 200), Arg(0, np.inf)], nan_ok=False, rtol=1e-12)

    def test_besselk_int(self):
        assert_mpmath_equal(sc.kn, mpmath.besselk, [IntArg(-200, 200), Arg(0, np.inf)], nan_ok=False, rtol=1e-12)

    def test_besselk_complex(self):
        assert_mpmath_equal(lambda v, z: sc.kv(v.real, z), exception_to_nan(lambda v, z: mpmath.besselk(v, z, **HYPERKW)), [Arg(-1e+100, 1e+100), ComplexArg()])

    def test_bessely(self):

        def mpbessely(v, x):
            r = float(mpmath.bessely(v, x, **HYPERKW))
            if abs(r) > 1e+305:
                r = np.inf * np.sign(r)
            if abs(r) == 0 and x == 0:
                return np.nan
            return r
        assert_mpmath_equal(sc.yv, exception_to_nan(mpbessely), [Arg(-1e+100, 1e+100), Arg(-100000000.0, 100000000.0)], n=5000)

    def test_bessely_complex(self):

        def mpbessely(v, x):
            r = complex(mpmath.bessely(v, x, **HYPERKW))
            if abs(r) > 1e+305:
                with np.errstate(invalid='ignore'):
                    r = np.inf * np.sign(r)
            return r
        assert_mpmath_equal(lambda v, z: sc.yv(v.real, z), exception_to_nan(mpbessely), [Arg(), ComplexArg()], n=15000)

    def test_bessely_int(self):

        def mpbessely(v, x):
            r = float(mpmath.bessely(v, x))
            if abs(r) == 0 and x == 0:
                return np.nan
            return r
        assert_mpmath_equal(lambda v, z: sc.yn(int(v), z), exception_to_nan(mpbessely), [IntArg(-1000, 1000), Arg(-100000000.0, 100000000.0)])

    def test_beta(self):
        bad_points = []

        def beta(a, b, nonzero=False):
            if a < -1000000000000.0 or b < -1000000000000.0:
                return np.nan
            if (a < 0 or b < 0) and abs(float(a + b)) % 1 == 0:
                if nonzero:
                    bad_points.append((float(a), float(b)))
                    return np.nan
            return mpmath.beta(a, b)
        assert_mpmath_equal(sc.beta, lambda a, b: beta(a, b, nonzero=True), [Arg(), Arg()], dps=400, ignore_inf_sign=True)
        assert_mpmath_equal(sc.beta, beta, np.array(bad_points), dps=400, ignore_inf_sign=True, atol=1e-11)

    def test_betainc(self):
        assert_mpmath_equal(sc.betainc, time_limited()(exception_to_nan(lambda a, b, x: mpmath.betainc(a, b, 0, x, regularized=True))), [Arg(), Arg(), Arg()])

    def test_betaincc(self):
        assert_mpmath_equal(sc.betaincc, time_limited()(exception_to_nan(lambda a, b, x: mpmath.betainc(a, b, x, 1, regularized=True))), [Arg(), Arg(), Arg()], dps=400)

    def test_binom(self):
        bad_points = []

        def binomial(n, k, nonzero=False):
            if abs(k) > 100000000.0 * (abs(n) + 1):
                return np.nan
            if n < k and abs(float(n - k) - np.round(float(n - k))) < 1e-15:
                if nonzero:
                    bad_points.append((float(n), float(k)))
                    return np.nan
            return mpmath.binomial(n, k)
        assert_mpmath_equal(sc.binom, lambda n, k: binomial(n, k, nonzero=True), [Arg(), Arg()], dps=400)
        assert_mpmath_equal(sc.binom, binomial, np.array(bad_points), dps=400, atol=1e-14)

    def test_chebyt_int(self):
        assert_mpmath_equal(lambda n, x: sc.eval_chebyt(int(n), x), exception_to_nan(lambda n, x: mpmath.chebyt(n, x, **HYPERKW)), [IntArg(), Arg()], dps=50)

    @pytest.mark.xfail(run=False, reason='some cases in hyp2f1 not fully accurate')
    def test_chebyt(self):
        assert_mpmath_equal(sc.eval_chebyt, lambda n, x: time_limited()(exception_to_nan(mpmath.chebyt))(n, x, **HYPERKW), [Arg(-101, 101), Arg()], n=10000)

    def test_chebyu_int(self):
        assert_mpmath_equal(lambda n, x: sc.eval_chebyu(int(n), x), exception_to_nan(lambda n, x: mpmath.chebyu(n, x, **HYPERKW)), [IntArg(), Arg()], dps=50)

    @pytest.mark.xfail(run=False, reason='some cases in hyp2f1 not fully accurate')
    def test_chebyu(self):
        assert_mpmath_equal(sc.eval_chebyu, lambda n, x: time_limited()(exception_to_nan(mpmath.chebyu))(n, x, **HYPERKW), [Arg(-101, 101), Arg()])

    def test_chi(self):

        def chi(x):
            return sc.shichi(x)[1]
        assert_mpmath_equal(chi, mpmath.chi, [Arg()])
        assert_mpmath_equal(chi, mpmath.chi, [FixedArg([88 - 1e-09, 88, 88 + 1e-09])])

    def test_chi_complex(self):

        def chi(z):
            return sc.shichi(z)[1]
        assert_mpmath_equal(chi, mpmath.chi, [ComplexArg(complex(-np.inf, -100000000.0), complex(np.inf, 100000000.0))], rtol=1e-12)

    def test_ci(self):

        def ci(x):
            return sc.sici(x)[1]
        assert_mpmath_equal(ci, mpmath.ci, [Arg(-100000000.0, 100000000.0)])

    def test_ci_complex(self):

        def ci(z):
            return sc.sici(z)[1]
        assert_mpmath_equal(ci, mpmath.ci, [ComplexArg(complex(-100000000.0, -np.inf), complex(100000000.0, np.inf))], rtol=1e-08)

    def test_cospi(self):
        eps = np.finfo(float).eps
        assert_mpmath_equal(_cospi, mpmath.cospi, [Arg()], nan_ok=False, rtol=2 * eps)

    def test_cospi_complex(self):
        assert_mpmath_equal(_cospi, mpmath.cospi, [ComplexArg()], nan_ok=False, rtol=1e-13)

    def test_digamma(self):
        assert_mpmath_equal(sc.digamma, exception_to_nan(mpmath.digamma), [Arg()], rtol=1e-12, dps=50)

    def test_digamma_complex(self):

        def param_filter(z):
            return np.where((z.real < 0) & (np.abs(z.imag) < 1.12), False, True)
        assert_mpmath_equal(sc.digamma, exception_to_nan(mpmath.digamma), [ComplexArg()], rtol=1e-13, dps=40, param_filter=param_filter)

    def test_e1(self):
        assert_mpmath_equal(sc.exp1, mpmath.e1, [Arg()], rtol=1e-14)

    def test_e1_complex(self):
        assert_mpmath_equal(sc.exp1, mpmath.e1, [ComplexArg(complex(-np.inf, -100000000.0), complex(np.inf, 100000000.0))], rtol=1e-11)
        assert_mpmath_equal(sc.exp1, mpmath.e1, (np.linspace(-50, 50, 171)[:, None] + np.r_[0, np.logspace(-3, 2, 61), -np.logspace(-3, 2, 11)] * 1j).ravel(), rtol=1e-11)
        assert_mpmath_equal(sc.exp1, mpmath.e1, np.linspace(-50, -35, 10000) + 0j, rtol=1e-11)

    def test_exprel(self):
        assert_mpmath_equal(sc.exprel, lambda x: mpmath.expm1(x) / x if x != 0 else mpmath.mpf('1.0'), [Arg(a=-np.log(np.finfo(np.float64).max), b=np.log(np.finfo(np.float64).max))])
        assert_mpmath_equal(sc.exprel, lambda x: mpmath.expm1(x) / x if x != 0 else mpmath.mpf('1.0'), np.array([1e-12, 1e-24, 0, 1000000000000.0, 1e+24, np.inf]), rtol=1e-11)
        assert_(np.isinf(sc.exprel(np.inf)))
        assert_(sc.exprel(-np.inf) == 0)

    def test_expm1_complex(self):
        assert_mpmath_equal(sc.expm1, mpmath.expm1, [ComplexArg(complex(-np.inf, -10000000.0), complex(np.inf, 10000000.0))])

    def test_log1p_complex(self):
        assert_mpmath_equal(sc.log1p, lambda x: mpmath.log(x + 1), [ComplexArg()], dps=60)

    def test_log1pmx(self):
        assert_mpmath_equal(_log1pmx, lambda x: mpmath.log(x + 1) - x, [Arg()], dps=60, rtol=1e-14)

    def test_ei(self):
        assert_mpmath_equal(sc.expi, mpmath.ei, [Arg()], rtol=1e-11)

    def test_ei_complex(self):
        assert_mpmath_equal(sc.expi, mpmath.ei, [ComplexArg(complex(-np.inf, -100000000.0), complex(np.inf, 100000000.0))], rtol=1e-09)

    def test_ellipe(self):
        assert_mpmath_equal(sc.ellipe, mpmath.ellipe, [Arg(b=1.0)])

    def test_ellipeinc(self):
        assert_mpmath_equal(sc.ellipeinc, mpmath.ellipe, [Arg(-1000.0, 1000.0), Arg(b=1.0)])

    def test_ellipeinc_largephi(self):
        assert_mpmath_equal(sc.ellipeinc, mpmath.ellipe, [Arg(), Arg()])

    def test_ellipf(self):
        assert_mpmath_equal(sc.ellipkinc, mpmath.ellipf, [Arg(-1000.0, 1000.0), Arg()])

    def test_ellipf_largephi(self):
        assert_mpmath_equal(sc.ellipkinc, mpmath.ellipf, [Arg(), Arg()])

    def test_ellipk(self):
        assert_mpmath_equal(sc.ellipk, mpmath.ellipk, [Arg(b=1.0)])
        assert_mpmath_equal(sc.ellipkm1, lambda m: mpmath.ellipk(1 - m), [Arg(a=0.0)], dps=400)

    def test_ellipkinc(self):

        def ellipkinc(phi, m):
            return mpmath.ellippi(0, phi, m)
        assert_mpmath_equal(sc.ellipkinc, ellipkinc, [Arg(-1000.0, 1000.0), Arg(b=1.0)], ignore_inf_sign=True)

    def test_ellipkinc_largephi(self):

        def ellipkinc(phi, m):
            return mpmath.ellippi(0, phi, m)
        assert_mpmath_equal(sc.ellipkinc, ellipkinc, [Arg(), Arg(b=1.0)], ignore_inf_sign=True)

    def test_ellipfun_sn(self):

        def sn(u, m):
            if u == 0:
                return 0
            else:
                return mpmath.ellipfun('sn', u=u, m=m)
        assert_mpmath_equal(lambda u, m: sc.ellipj(u, m)[0], sn, [Arg(-1000000.0, 1000000.0), Arg(a=0, b=1)], rtol=1e-08)

    def test_ellipfun_cn(self):
        assert_mpmath_equal(lambda u, m: sc.ellipj(u, m)[1], lambda u, m: mpmath.ellipfun('cn', u=u, m=m), [Arg(-1000000.0, 1000000.0), Arg(a=0, b=1)], rtol=1e-08)

    def test_ellipfun_dn(self):
        assert_mpmath_equal(lambda u, m: sc.ellipj(u, m)[2], lambda u, m: mpmath.ellipfun('dn', u=u, m=m), [Arg(-1000000.0, 1000000.0), Arg(a=0, b=1)], rtol=1e-08)

    def test_erf(self):
        assert_mpmath_equal(sc.erf, lambda z: mpmath.erf(z), [Arg()])

    def test_erf_complex(self):
        assert_mpmath_equal(sc.erf, lambda z: mpmath.erf(z), [ComplexArg()], n=200)

    def test_erfc(self):
        assert_mpmath_equal(sc.erfc, exception_to_nan(lambda z: mpmath.erfc(z)), [Arg()], rtol=1e-13)

    def test_erfc_complex(self):
        assert_mpmath_equal(sc.erfc, exception_to_nan(lambda z: mpmath.erfc(z)), [ComplexArg()], n=200)

    def test_erfi(self):
        assert_mpmath_equal(sc.erfi, mpmath.erfi, [Arg()], n=200)

    def test_erfi_complex(self):
        assert_mpmath_equal(sc.erfi, mpmath.erfi, [ComplexArg()], n=200)

    def test_ndtr(self):
        assert_mpmath_equal(sc.ndtr, exception_to_nan(lambda z: mpmath.ncdf(z)), [Arg()], n=200)

    def test_ndtr_complex(self):
        assert_mpmath_equal(sc.ndtr, lambda z: mpmath.erfc(-z / np.sqrt(2.0)) / 2.0, [ComplexArg(a=complex(-10000, -10000), b=complex(10000, 10000))], n=400)

    def test_log_ndtr(self):
        assert_mpmath_equal(sc.log_ndtr, exception_to_nan(lambda z: mpmath.log(mpmath.ncdf(z))), [Arg()], n=600, dps=300, rtol=1e-13)

    def test_log_ndtr_complex(self):
        assert_mpmath_equal(sc.log_ndtr, exception_to_nan(lambda z: mpmath.log(mpmath.erfc(-z / np.sqrt(2.0)) / 2.0)), [ComplexArg(a=complex(-10000, -100), b=complex(10000, 100))], n=200, dps=300)

    def test_eulernum(self):
        assert_mpmath_equal(lambda n: sc.euler(n)[-1], mpmath.eulernum, [IntArg(1, 10000)], n=10000)

    def test_expint(self):
        assert_mpmath_equal(sc.expn, mpmath.expint, [IntArg(0, 200), Arg(0, np.inf)], rtol=1e-13, dps=160)

    def test_fresnels(self):

        def fresnels(x):
            return sc.fresnel(x)[0]
        assert_mpmath_equal(fresnels, mpmath.fresnels, [Arg()])

    def test_fresnelc(self):

        def fresnelc(x):
            return sc.fresnel(x)[1]
        assert_mpmath_equal(fresnelc, mpmath.fresnelc, [Arg()])

    def test_gamma(self):
        assert_mpmath_equal(sc.gamma, exception_to_nan(mpmath.gamma), [Arg()])

    def test_gamma_complex(self):
        assert_mpmath_equal(sc.gamma, exception_to_nan(mpmath.gamma), [ComplexArg()], rtol=5e-13)

    def test_gammainc(self):
        assert_mpmath_equal(sc.gammainc, lambda z, b: mpmath.gammainc(z, b=b, regularized=True), [Arg(0, 10000.0, inclusive_a=False), Arg(0, 10000.0)], nan_ok=False, rtol=1e-11)

    def test_gammaincc(self):
        assert_mpmath_equal(sc.gammaincc, lambda z, a: mpmath.gammainc(z, a=a, regularized=True), [Arg(0, 10000.0, inclusive_a=False), Arg(0, 10000.0)], nan_ok=False, rtol=1e-11)

    def test_gammaln(self):

        def f(z):
            return mpmath.loggamma(z).real
        assert_mpmath_equal(sc.gammaln, exception_to_nan(f), [Arg()])

    @pytest.mark.xfail(run=False)
    def test_gegenbauer(self):
        assert_mpmath_equal(sc.eval_gegenbauer, exception_to_nan(mpmath.gegenbauer), [Arg(-1000.0, 1000.0), Arg(), Arg()])

    def test_gegenbauer_int(self):

        def gegenbauer(n, a, x):
            if abs(a) > 1e+100:
                return np.nan
            if n == 0:
                r = 1.0
            elif n == 1:
                r = 2 * a * x
            else:
                r = mpmath.gegenbauer(n, a, x)
            if float(r) == 0 and a < -1 and (float(a) == int(float(a))):
                r = mpmath.gegenbauer(n, a + mpmath.mpf('1e-50'), x)
                if abs(r) < mpmath.mpf('1e-50'):
                    r = mpmath.mpf('0.0')
            if abs(r) > 1e+270:
                return np.inf
            return r

        def sc_gegenbauer(n, a, x):
            r = sc.eval_gegenbauer(int(n), a, x)
            if abs(r) > 1e+270:
                return np.inf
            return r
        assert_mpmath_equal(sc_gegenbauer, exception_to_nan(gegenbauer), [IntArg(0, 100), Arg(-1000000000.0, 1000000000.0), Arg()], n=40000, dps=100, ignore_inf_sign=True, rtol=1e-06)
        assert_mpmath_equal(sc_gegenbauer, exception_to_nan(gegenbauer), [IntArg(0, 100), Arg(), FixedArg(np.logspace(-30, -4, 30))], dps=100, ignore_inf_sign=True)

    @pytest.mark.xfail(run=False)
    def test_gegenbauer_complex(self):
        assert_mpmath_equal(lambda n, a, x: sc.eval_gegenbauer(int(n), a.real, x), exception_to_nan(mpmath.gegenbauer), [IntArg(0, 100), Arg(), ComplexArg()])

    @nonfunctional_tooslow
    def test_gegenbauer_complex_general(self):
        assert_mpmath_equal(lambda n, a, x: sc.eval_gegenbauer(n.real, a.real, x), exception_to_nan(mpmath.gegenbauer), [Arg(-1000.0, 1000.0), Arg(), ComplexArg()])

    def test_hankel1(self):
        assert_mpmath_equal(sc.hankel1, exception_to_nan(lambda v, x: mpmath.hankel1(v, x, **HYPERKW)), [Arg(-1e+20, 1e+20), Arg()])

    def test_hankel2(self):
        assert_mpmath_equal(sc.hankel2, exception_to_nan(lambda v, x: mpmath.hankel2(v, x, **HYPERKW)), [Arg(-1e+20, 1e+20), Arg()])

    @pytest.mark.xfail(run=False, reason='issues at intermediately large orders')
    def test_hermite(self):
        assert_mpmath_equal(lambda n, x: sc.eval_hermite(int(n), x), exception_to_nan(mpmath.hermite), [IntArg(0, 10000), Arg()])

    def test_hyp0f1(self):
        KW = dict(maxprec=400, maxterms=1500)
        assert_mpmath_equal(sc.hyp0f1, lambda a, x: mpmath.hyp0f1(a, x, **KW), [Arg(-10000000.0, 10000000.0), Arg(0, 100000.0)], n=5000)

    def test_hyp0f1_complex(self):
        assert_mpmath_equal(lambda a, z: sc.hyp0f1(a.real, z), exception_to_nan(lambda a, x: mpmath.hyp0f1(a, x, **HYPERKW)), [Arg(-10, 10), ComplexArg(complex(-120, -120), complex(120, 120))])

    def test_hyp1f1(self):

        def mpmath_hyp1f1(a, b, x):
            try:
                return mpmath.hyp1f1(a, b, x)
            except ZeroDivisionError:
                return np.inf
        assert_mpmath_equal(sc.hyp1f1, mpmath_hyp1f1, [Arg(-50, 50), Arg(1, 50, inclusive_a=False), Arg(-50, 50)], n=500, nan_ok=False)

    @pytest.mark.xfail(run=False)
    def test_hyp1f1_complex(self):
        assert_mpmath_equal(inf_to_nan(lambda a, b, x: sc.hyp1f1(a.real, b.real, x)), exception_to_nan(lambda a, b, x: mpmath.hyp1f1(a, b, x, **HYPERKW)), [Arg(-1000.0, 1000.0), Arg(-1000.0, 1000.0), ComplexArg()], n=2000)

    @nonfunctional_tooslow
    def test_hyp2f1_complex(self):
        assert_mpmath_equal(lambda a, b, c, x: sc.hyp2f1(a.real, b.real, c.real, x), exception_to_nan(lambda a, b, c, x: mpmath.hyp2f1(a, b, c, x, **HYPERKW)), [Arg(-100.0, 100.0), Arg(-100.0, 100.0), Arg(-100.0, 100.0), ComplexArg()], n=10)

    @pytest.mark.xfail(run=False)
    def test_hyperu(self):
        assert_mpmath_equal(sc.hyperu, exception_to_nan(lambda a, b, x: mpmath.hyperu(a, b, x, **HYPERKW)), [Arg(), Arg(), Arg()])

    @pytest.mark.xfail_on_32bit('mpmath issue gh-342: unsupported operand mpz, long for pow')
    def test_igam_fac(self):

        def mp_igam_fac(a, x):
            return mpmath.power(x, a) * mpmath.exp(-x) / mpmath.gamma(a)
        assert_mpmath_equal(_igam_fac, mp_igam_fac, [Arg(0, 100000000000000.0, inclusive_a=False), Arg(0, 100000000000000.0)], rtol=1e-10)

    def test_j0(self):
        assert_mpmath_equal(sc.j0, mpmath.j0, [Arg(-1000.0, 1000.0)])
        assert_mpmath_equal(sc.j0, mpmath.j0, [Arg(-100000000.0, 100000000.0)], rtol=1e-05)

    def test_j1(self):
        assert_mpmath_equal(sc.j1, mpmath.j1, [Arg(-1000.0, 1000.0)])
        assert_mpmath_equal(sc.j1, mpmath.j1, [Arg(-100000000.0, 100000000.0)], rtol=1e-05)

    @pytest.mark.xfail(run=False)
    def test_jacobi(self):
        assert_mpmath_equal(sc.eval_jacobi, exception_to_nan(lambda a, b, c, x: mpmath.jacobi(a, b, c, x, **HYPERKW)), [Arg(), Arg(), Arg(), Arg()])
        assert_mpmath_equal(lambda n, b, c, x: sc.eval_jacobi(int(n), b, c, x), exception_to_nan(lambda a, b, c, x: mpmath.jacobi(a, b, c, x, **HYPERKW)), [IntArg(), Arg(), Arg(), Arg()])

    def test_jacobi_int(self):

        def jacobi(n, a, b, x):
            if n == 0:
                return 1.0
            return mpmath.jacobi(n, a, b, x)
        assert_mpmath_equal(lambda n, a, b, x: sc.eval_jacobi(int(n), a, b, x), lambda n, a, b, x: exception_to_nan(jacobi)(n, a, b, x, **HYPERKW), [IntArg(), Arg(), Arg(), Arg()], n=20000, dps=50)

    def test_kei(self):

        def kei(x):
            if x == 0:
                return -pi / 4
            return exception_to_nan(mpmath.kei)(0, x, **HYPERKW)
        assert_mpmath_equal(sc.kei, kei, [Arg(-1e+30, 1e+30)], n=1000)

    def test_ker(self):
        assert_mpmath_equal(sc.ker, exception_to_nan(lambda x: mpmath.ker(0, x, **HYPERKW)), [Arg(-1e+30, 1e+30)], n=1000)

    @nonfunctional_tooslow
    def test_laguerre(self):
        assert_mpmath_equal(trace_args(sc.eval_laguerre), lambda n, x: exception_to_nan(mpmath.laguerre)(n, x, **HYPERKW), [Arg(), Arg()])

    def test_laguerre_int(self):
        assert_mpmath_equal(lambda n, x: sc.eval_laguerre(int(n), x), lambda n, x: exception_to_nan(mpmath.laguerre)(n, x, **HYPERKW), [IntArg(), Arg()], n=20000)

    @pytest.mark.xfail_on_32bit('see gh-3551 for bad points')
    def test_lambertw_real(self):
        assert_mpmath_equal(lambda x, k: sc.lambertw(x, int(k.real)), lambda x, k: mpmath.lambertw(x, int(k.real)), [ComplexArg(-np.inf, np.inf), IntArg(0, 10)], rtol=1e-13, nan_ok=False)

    def test_lanczos_sum_expg_scaled(self):
        maxgamma = 171.6243769563027
        e = np.exp(1)
        g = 6.02468004077673

        def gamma(x):
            with np.errstate(over='ignore'):
                fac = ((x + g - 0.5) / e) ** (x - 0.5)
                if fac != np.inf:
                    res = fac * _lanczos_sum_expg_scaled(x)
                else:
                    fac = ((x + g - 0.5) / e) ** (0.5 * (x - 0.5))
                    res = fac * _lanczos_sum_expg_scaled(x)
                    res *= fac
            return res
        assert_mpmath_equal(gamma, mpmath.gamma, [Arg(0, maxgamma, inclusive_a=False)], rtol=1e-13)

    @nonfunctional_tooslow
    def test_legendre(self):
        assert_mpmath_equal(sc.eval_legendre, mpmath.legendre, [Arg(), Arg()])

    def test_legendre_int(self):
        assert_mpmath_equal(lambda n, x: sc.eval_legendre(int(n), x), lambda n, x: exception_to_nan(mpmath.legendre)(n, x, **HYPERKW), [IntArg(), Arg()], n=20000)
        assert_mpmath_equal(lambda n, x: sc.eval_legendre(int(n), x), lambda n, x: exception_to_nan(mpmath.legendre)(n, x, **HYPERKW), [IntArg(), FixedArg(np.logspace(-30, -4, 20))])

    def test_legenp(self):

        def lpnm(n, m, z):
            try:
                v = sc.lpmn(m, n, z)[0][-1, -1]
            except ValueError:
                return np.nan
            if abs(v) > 1e+306:
                v = np.inf * np.sign(v.real)
            return v

        def lpnm_2(n, m, z):
            v = sc.lpmv(m, n, z)
            if abs(v) > 1e+306:
                v = np.inf * np.sign(v.real)
            return v

        def legenp(n, m, z):
            if (z == 1 or z == -1) and int(n) == n:
                if m == 0:
                    if n < 0:
                        n = -n - 1
                    return mpmath.power(mpmath.sign(z), n)
                else:
                    return 0
            if abs(z) < 1e-15:
                return np.nan
            typ = 2 if abs(z) < 1 else 3
            v = exception_to_nan(mpmath.legenp)(n, m, z, type=typ)
            if abs(v) > 1e+306:
                v = mpmath.inf * mpmath.sign(v.real)
            return v
        assert_mpmath_equal(lpnm, legenp, [IntArg(-100, 100), IntArg(-100, 100), Arg()])
        assert_mpmath_equal(lpnm_2, legenp, [IntArg(-100, 100), Arg(-100, 100), Arg(-1, 1)], atol=1e-10)

    def test_legenp_complex_2(self):

        def clpnm(n, m, z):
            try:
                return sc.clpmn(m.real, n.real, z, type=2)[0][-1, -1]
            except ValueError:
                return np.nan

        def legenp(n, m, z):
            if abs(z) < 1e-15:
                return np.nan
            return exception_to_nan(mpmath.legenp)(int(n.real), int(m.real), z, type=2)
        x = np.array([-2, -0.99, -0.5, 0, 1e-05, 0.5, 0.99, 20, 2000.0])
        y = np.array([-1000.0, -0.5, 0.5, 1.3])
        z = (x[:, None] + 1j * y[None, :]).ravel()
        assert_mpmath_equal(clpnm, legenp, [FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg(z)], rtol=1e-06, n=500)

    def test_legenp_complex_3(self):

        def clpnm(n, m, z):
            try:
                return sc.clpmn(m.real, n.real, z, type=3)[0][-1, -1]
            except ValueError:
                return np.nan

        def legenp(n, m, z):
            if abs(z) < 1e-15:
                return np.nan
            return exception_to_nan(mpmath.legenp)(int(n.real), int(m.real), z, type=3)
        x = np.array([-2, -0.99, -0.5, 0, 1e-05, 0.5, 0.99, 20, 2000.0])
        y = np.array([-1000.0, -0.5, 0.5, 1.3])
        z = (x[:, None] + 1j * y[None, :]).ravel()
        assert_mpmath_equal(clpnm, legenp, [FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg(z)], rtol=1e-06, n=500)

    @pytest.mark.xfail(run=False, reason='apparently picks wrong function at |z| > 1')
    def test_legenq(self):

        def lqnm(n, m, z):
            return sc.lqmn(m, n, z)[0][-1, -1]

        def legenq(n, m, z):
            if abs(z) < 1e-15:
                return np.nan
            return exception_to_nan(mpmath.legenq)(n, m, z, type=2)
        assert_mpmath_equal(lqnm, legenq, [IntArg(0, 100), IntArg(0, 100), Arg()])

    @nonfunctional_tooslow
    def test_legenq_complex(self):

        def lqnm(n, m, z):
            return sc.lqmn(int(m.real), int(n.real), z)[0][-1, -1]

        def legenq(n, m, z):
            if abs(z) < 1e-15:
                return np.nan
            return exception_to_nan(mpmath.legenq)(int(n.real), int(m.real), z, type=2)
        assert_mpmath_equal(lqnm, legenq, [IntArg(0, 100), IntArg(0, 100), ComplexArg()], n=100)

    def test_lgam1p(self):

        def param_filter(x):
            return np.where((np.floor(x) == x) & (x <= 0), False, True)

        def mp_lgam1p(z):
            return mpmath.loggamma(1 + z).real
        assert_mpmath_equal(_lgam1p, mp_lgam1p, [Arg()], rtol=1e-13, dps=100, param_filter=param_filter)

    def test_loggamma(self):

        def mpmath_loggamma(z):
            try:
                res = mpmath.loggamma(z)
            except ValueError:
                res = complex(np.nan, np.nan)
            return res
        assert_mpmath_equal(sc.loggamma, mpmath_loggamma, [ComplexArg()], nan_ok=False, distinguish_nan_and_inf=False, rtol=5e-14)

    @pytest.mark.xfail(run=False)
    def test_pcfd(self):

        def pcfd(v, x):
            return sc.pbdv(v, x)[0]
        assert_mpmath_equal(pcfd, exception_to_nan(lambda v, x: mpmath.pcfd(v, x, **HYPERKW)), [Arg(), Arg()])

    @pytest.mark.xfail(run=False, reason="it's not the same as the mpmath function --- maybe different definition?")
    def test_pcfv(self):

        def pcfv(v, x):
            return sc.pbvv(v, x)[0]
        assert_mpmath_equal(pcfv, lambda v, x: time_limited()(exception_to_nan(mpmath.pcfv))(v, x, **HYPERKW), [Arg(), Arg()], n=1000)

    def test_pcfw(self):

        def pcfw(a, x):
            return sc.pbwa(a, x)[0]

        def dpcfw(a, x):
            return sc.pbwa(a, x)[1]

        def mpmath_dpcfw(a, x):
            return mpmath.diff(mpmath.pcfw, (a, x), (0, 1))
        assert_mpmath_equal(pcfw, mpmath.pcfw, [Arg(-5, 5), Arg(-5, 5)], rtol=2e-08, n=100)
        assert_mpmath_equal(dpcfw, mpmath_dpcfw, [Arg(-5, 5), Arg(-5, 5)], rtol=2e-09, n=100)

    @pytest.mark.xfail(run=False, reason='issues at large arguments (atol OK, rtol not) and <eps-close to z=0')
    def test_polygamma(self):
        assert_mpmath_equal(sc.polygamma, time_limited()(exception_to_nan(mpmath.polygamma)), [IntArg(0, 1000), Arg()])

    def test_rgamma(self):
        assert_mpmath_equal(sc.rgamma, mpmath.rgamma, [Arg(-8000, np.inf)], n=5000, nan_ok=False, ignore_inf_sign=True)

    def test_rgamma_complex(self):
        assert_mpmath_equal(sc.rgamma, exception_to_nan(mpmath.rgamma), [ComplexArg()], rtol=5e-13)

    @pytest.mark.xfail(reason='see gh-3551 for bad points on 32 bit systems and gh-8095 for another bad point')
    def test_rf(self):
        if _pep440.parse(mpmath.__version__) >= _pep440.Version('1.0.0'):
            mppoch = mpmath.rf
        else:

            def mppoch(a, m):
                if float(a + m) == int(a + m) and float(a + m) <= 0:
                    a = mpmath.mpf(a)
                    m = int(a + m) - a
                return mpmath.rf(a, m)
        assert_mpmath_equal(sc.poch, mppoch, [Arg(), Arg()], dps=400)

    def test_sinpi(self):
        eps = np.finfo(float).eps
        assert_mpmath_equal(_sinpi, mpmath.sinpi, [Arg()], nan_ok=False, rtol=2 * eps)

    def test_sinpi_complex(self):
        assert_mpmath_equal(_sinpi, mpmath.sinpi, [ComplexArg()], nan_ok=False, rtol=2e-14)

    def test_shi(self):

        def shi(x):
            return sc.shichi(x)[0]
        assert_mpmath_equal(shi, mpmath.shi, [Arg()])
        assert_mpmath_equal(shi, mpmath.shi, [FixedArg([88 - 1e-09, 88, 88 + 1e-09])])

    def test_shi_complex(self):

        def shi(z):
            return sc.shichi(z)[0]
        assert_mpmath_equal(shi, mpmath.shi, [ComplexArg(complex(-np.inf, -100000000.0), complex(np.inf, 100000000.0))], rtol=1e-12)

    def test_si(self):

        def si(x):
            return sc.sici(x)[0]
        assert_mpmath_equal(si, mpmath.si, [Arg()])

    def test_si_complex(self):

        def si(z):
            return sc.sici(z)[0]
        assert_mpmath_equal(si, mpmath.si, [ComplexArg(complex(-100000000.0, -np.inf), complex(100000000.0, np.inf))], rtol=1e-12)

    def test_spence(self):

        def dilog(x):
            return mpmath.polylog(2, 1 - x)
        assert_mpmath_equal(sc.spence, exception_to_nan(dilog), [Arg(0, np.inf)], rtol=1e-14)

    def test_spence_complex(self):

        def dilog(z):
            return mpmath.polylog(2, 1 - z)
        assert_mpmath_equal(sc.spence, exception_to_nan(dilog), [ComplexArg()], rtol=1e-14)

    def test_spherharm(self):

        def spherharm(l, m, theta, phi):
            if m > l:
                return np.nan
            return sc.sph_harm(m, l, phi, theta)
        assert_mpmath_equal(spherharm, mpmath.spherharm, [IntArg(0, 100), IntArg(0, 100), Arg(a=0, b=pi), Arg(a=0, b=2 * pi)], atol=1e-08, n=6000, dps=150)

    def test_struveh(self):
        assert_mpmath_equal(sc.struve, exception_to_nan(mpmath.struveh), [Arg(-10000.0, 10000.0), Arg(0, 10000.0)], rtol=5e-10)

    def test_struvel(self):

        def mp_struvel(v, z):
            if v < 0 and z < -v and (abs(v) > 1000):
                old_dps = mpmath.mp.dps
                try:
                    mpmath.mp.dps = 300
                    return mpmath.struvel(v, z)
                finally:
                    mpmath.mp.dps = old_dps
            return mpmath.struvel(v, z)
        assert_mpmath_equal(sc.modstruve, exception_to_nan(mp_struvel), [Arg(-10000.0, 10000.0), Arg(0, 10000.0)], rtol=5e-10, ignore_inf_sign=True)

    def test_wrightomega_real(self):

        def mpmath_wrightomega_real(x):
            return mpmath.lambertw(mpmath.exp(x), mpmath.mpf('-0.5'))
        assert_mpmath_equal(sc.wrightomega, mpmath_wrightomega_real, [Arg(-1000, 1e+21)], rtol=5e-15, atol=0, nan_ok=False)

    def test_wrightomega(self):
        assert_mpmath_equal(sc.wrightomega, lambda z: _mpmath_wrightomega(z, 25), [ComplexArg()], rtol=1e-14, nan_ok=False)

    def test_hurwitz_zeta(self):
        assert_mpmath_equal(sc.zeta, exception_to_nan(mpmath.zeta), [Arg(a=1, b=10000000000.0, inclusive_a=False), Arg(a=0, inclusive_a=False)])

    def test_riemann_zeta(self):
        assert_mpmath_equal(sc.zeta, lambda x: mpmath.zeta(x) if x != 1 else mpmath.inf, [Arg(-100, 100)], nan_ok=False, rtol=5e-13)

    def test_zetac(self):
        assert_mpmath_equal(sc.zetac, lambda x: mpmath.zeta(x) - 1 if x != 1 else mpmath.inf, [Arg(-100, 100)], nan_ok=False, dps=45, rtol=5e-13)

    def test_boxcox(self):

        def mp_boxcox(x, lmbda):
            x = mpmath.mp.mpf(x)
            lmbda = mpmath.mp.mpf(lmbda)
            if lmbda == 0:
                return mpmath.mp.log(x)
            else:
                return mpmath.mp.powm1(x, lmbda) / lmbda
        assert_mpmath_equal(sc.boxcox, exception_to_nan(mp_boxcox), [Arg(a=0, inclusive_a=False), Arg()], n=200, dps=60, rtol=1e-13)

    def test_boxcox1p(self):

        def mp_boxcox1p(x, lmbda):
            x = mpmath.mp.mpf(x)
            lmbda = mpmath.mp.mpf(lmbda)
            one = mpmath.mp.mpf(1)
            if lmbda == 0:
                return mpmath.mp.log(one + x)
            else:
                return mpmath.mp.powm1(one + x, lmbda) / lmbda
        assert_mpmath_equal(sc.boxcox1p, exception_to_nan(mp_boxcox1p), [Arg(a=-1, inclusive_a=False), Arg()], n=200, dps=60, rtol=1e-13)

    def test_spherical_jn(self):

        def mp_spherical_jn(n, z):
            arg = mpmath.mpmathify(z)
            out = mpmath.besselj(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_jn(int(n), z), exception_to_nan(mp_spherical_jn), [IntArg(0, 200), Arg(-100000000.0, 100000000.0)], dps=300)

    def test_spherical_jn_complex(self):

        def mp_spherical_jn(n, z):
            arg = mpmath.mpmathify(z)
            out = mpmath.besselj(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_jn(int(n.real), z), exception_to_nan(mp_spherical_jn), [IntArg(0, 200), ComplexArg()])

    def test_spherical_yn(self):

        def mp_spherical_yn(n, z):
            arg = mpmath.mpmathify(z)
            out = mpmath.bessely(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_yn(int(n), z), exception_to_nan(mp_spherical_yn), [IntArg(0, 200), Arg(-10000000000.0, 10000000000.0)], dps=100)

    def test_spherical_yn_complex(self):

        def mp_spherical_yn(n, z):
            arg = mpmath.mpmathify(z)
            out = mpmath.bessely(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_yn(int(n.real), z), exception_to_nan(mp_spherical_yn), [IntArg(0, 200), ComplexArg()])

    def test_spherical_in(self):

        def mp_spherical_in(n, z):
            arg = mpmath.mpmathify(z)
            out = mpmath.besseli(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_in(int(n), z), exception_to_nan(mp_spherical_in), [IntArg(0, 200), Arg()], dps=200, atol=10 ** (-278))

    def test_spherical_in_complex(self):

        def mp_spherical_in(n, z):
            arg = mpmath.mpmathify(z)
            out = mpmath.besseli(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_in(int(n.real), z), exception_to_nan(mp_spherical_in), [IntArg(0, 200), ComplexArg()])

    def test_spherical_kn(self):

        def mp_spherical_kn(n, z):
            out = mpmath.besselk(n + mpmath.mpf(1) / 2, z) * mpmath.sqrt(mpmath.pi / (2 * mpmath.mpmathify(z)))
            if mpmath.mpmathify(z).imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_kn(int(n), z), exception_to_nan(mp_spherical_kn), [IntArg(0, 150), Arg()], dps=100)

    @pytest.mark.xfail(run=False, reason='Accuracy issues near z = -1 inherited from kv.')
    def test_spherical_kn_complex(self):

        def mp_spherical_kn(n, z):
            arg = mpmath.mpmathify(z)
            out = mpmath.besselk(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_kn(int(n.real), z), exception_to_nan(mp_spherical_kn), [IntArg(0, 200), ComplexArg()], dps=200)