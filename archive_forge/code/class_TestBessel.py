import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
class TestBessel:

    def test_itj0y0(self):
        it0 = array(special.itj0y0(0.2))
        assert_array_almost_equal(it0, array([0.19933433254006822, -0.34570883800412566]), 8)

    def test_it2j0y0(self):
        it2 = array(special.it2j0y0(0.2))
        assert_array_almost_equal(it2, array([0.004993754627460186, -0.43423067011231614]), 8)

    def test_negv_iv(self):
        assert_equal(special.iv(3, 2), special.iv(-3, 2))

    def test_j0(self):
        oz = special.j0(0.1)
        ozr = special.jn(0, 0.1)
        assert_almost_equal(oz, ozr, 8)

    def test_j1(self):
        o1 = special.j1(0.1)
        o1r = special.jn(1, 0.1)
        assert_almost_equal(o1, o1r, 8)

    def test_jn(self):
        jnnr = special.jn(1, 0.2)
        assert_almost_equal(jnnr, 0.099500832639236, 8)

    def test_negv_jv(self):
        assert_almost_equal(special.jv(-3, 2), -special.jv(3, 2), 14)

    def test_jv(self):
        values = [[0, 0.1, 0.99750156206604], [2.0 / 3, 1e-08, 3.239028506761532e-06], [2.0 / 3, 1e-10, 1.503423854873779e-07], [3.1, 1e-10, 1.711956265409013e-33], [2.0 / 3, 4.0, -0.2325440850267039]]
        for i, (v, x, y) in enumerate(values):
            yc = special.jv(v, x)
            assert_almost_equal(yc, y, 8, err_msg='test #%d' % i)

    def test_negv_jve(self):
        assert_almost_equal(special.jve(-3, 2), -special.jve(3, 2), 14)

    def test_jve(self):
        jvexp = special.jve(1, 0.2)
        assert_almost_equal(jvexp, 0.099500832639236, 8)
        jvexp1 = special.jve(1, 0.2 + 1j)
        z = 0.2 + 1j
        jvexpr = special.jv(1, z) * exp(-abs(z.imag))
        assert_almost_equal(jvexp1, jvexpr, 8)

    def test_jn_zeros(self):
        jn0 = special.jn_zeros(0, 5)
        jn1 = special.jn_zeros(1, 5)
        assert_array_almost_equal(jn0, array([2.4048255577, 5.5200781103, 8.6537279129, 11.7915344391, 14.9309177086]), 4)
        assert_array_almost_equal(jn1, array([3.83171, 7.01559, 10.17347, 13.32369, 16.47063]), 4)
        jn102 = special.jn_zeros(102, 5)
        assert_allclose(jn102, array([110.8917493599204, 117.83464175788309, 123.70194191713507, 129.02417238949093, 134.00114761868423]), rtol=1e-13)
        jn301 = special.jn_zeros(301, 5)
        assert_allclose(jn301, array([313.5909786669883, 323.2154977609629, 331.2233873865675, 338.39676338872084, 345.03284233056064]), rtol=1e-13)

    def test_jn_zeros_slow(self):
        jn0 = special.jn_zeros(0, 300)
        assert_allclose(jn0[260 - 1], 816.0288449506887, rtol=1e-13)
        assert_allclose(jn0[280 - 1], 878.8606870712442, rtol=1e-13)
        assert_allclose(jn0[300 - 1], 941.6925306531796, rtol=1e-13)
        jn10 = special.jn_zeros(10, 300)
        assert_allclose(jn10[260 - 1], 831.6766851430563, rtol=1e-13)
        assert_allclose(jn10[280 - 1], 894.5127509537132, rtol=1e-13)
        assert_allclose(jn10[300 - 1], 957.3482637086654, rtol=1e-13)
        jn3010 = special.jn_zeros(3010, 5)
        assert_allclose(jn3010, array([3036.86590780927, 3057.06598526482, 3073.66360690272, 3088.37736494778, 3101.86438139042]), rtol=1e-08)

    def test_jnjnp_zeros(self):
        jn = special.jn

        def jnp(n, x):
            return (jn(n - 1, x) - jn(n + 1, x)) / 2
        for nt in range(1, 30):
            z, n, m, t = special.jnjnp_zeros(nt)
            for zz, nn, tt in zip(z, n, t):
                if tt == 0:
                    assert_allclose(jn(nn, zz), 0, atol=1e-06)
                elif tt == 1:
                    assert_allclose(jnp(nn, zz), 0, atol=1e-06)
                else:
                    raise AssertionError('Invalid t return for nt=%d' % nt)

    def test_jnp_zeros(self):
        jnp = special.jnp_zeros(1, 5)
        assert_array_almost_equal(jnp, array([1.84118, 5.33144, 8.53632, 11.706, 14.86359]), 4)
        jnp = special.jnp_zeros(443, 5)
        assert_allclose(special.jvp(443, jnp), 0, atol=1e-15)

    def test_jnyn_zeros(self):
        jnz = special.jnyn_zeros(1, 5)
        assert_array_almost_equal(jnz, (array([3.83171, 7.01559, 10.17347, 13.32369, 16.47063]), array([1.84118, 5.33144, 8.53632, 11.706, 14.86359]), array([2.19714, 5.42968, 8.59601, 11.74915, 14.89744]), array([3.68302, 6.9415, 10.1234, 13.28576, 16.44006])), 5)

    def test_jvp(self):
        jvprim = special.jvp(2, 2)
        jv0 = (special.jv(1, 2) - special.jv(3, 2)) / 2
        assert_almost_equal(jvprim, jv0, 10)

    def test_k0(self):
        ozk = special.k0(0.1)
        ozkr = special.kv(0, 0.1)
        assert_almost_equal(ozk, ozkr, 8)

    def test_k0e(self):
        ozke = special.k0e(0.1)
        ozker = special.kve(0, 0.1)
        assert_almost_equal(ozke, ozker, 8)

    def test_k1(self):
        o1k = special.k1(0.1)
        o1kr = special.kv(1, 0.1)
        assert_almost_equal(o1k, o1kr, 8)

    def test_k1e(self):
        o1ke = special.k1e(0.1)
        o1ker = special.kve(1, 0.1)
        assert_almost_equal(o1ke, o1ker, 8)

    def test_jacobi(self):
        a = 5 * np.random.random() - 1
        b = 5 * np.random.random() - 1
        P0 = special.jacobi(0, a, b)
        P1 = special.jacobi(1, a, b)
        P2 = special.jacobi(2, a, b)
        P3 = special.jacobi(3, a, b)
        assert_array_almost_equal(P0.c, [1], 13)
        assert_array_almost_equal(P1.c, array([a + b + 2, a - b]) / 2.0, 13)
        cp = [(a + b + 3) * (a + b + 4), 4 * (a + b + 3) * (a + 2), 4 * (a + 1) * (a + 2)]
        p2c = [cp[0], cp[1] - 2 * cp[0], cp[2] - cp[1] + cp[0]]
        assert_array_almost_equal(P2.c, array(p2c) / 8.0, 13)
        cp = [(a + b + 4) * (a + b + 5) * (a + b + 6), 6 * (a + b + 4) * (a + b + 5) * (a + 3), 12 * (a + b + 4) * (a + 2) * (a + 3), 8 * (a + 1) * (a + 2) * (a + 3)]
        p3c = [cp[0], cp[1] - 3 * cp[0], cp[2] - 2 * cp[1] + 3 * cp[0], cp[3] - cp[2] + cp[1] - cp[0]]
        assert_array_almost_equal(P3.c, array(p3c) / 48.0, 13)

    def test_kn(self):
        kn1 = special.kn(0, 0.2)
        assert_almost_equal(kn1, 1.7527038555281462, 8)

    def test_negv_kv(self):
        assert_equal(special.kv(3.0, 2.2), special.kv(-3.0, 2.2))

    def test_kv0(self):
        kv0 = special.kv(0, 0.2)
        assert_almost_equal(kv0, 1.7527038555281462, 10)

    def test_kv1(self):
        kv1 = special.kv(1, 0.2)
        assert_almost_equal(kv1, 4.775972543220472, 10)

    def test_kv2(self):
        kv2 = special.kv(2, 0.2)
        assert_almost_equal(kv2, 49.51242928773287, 10)

    def test_kn_largeorder(self):
        assert_allclose(special.kn(32, 1), 1.751659666457429e+43)

    def test_kv_largearg(self):
        assert_equal(special.kv(0, 1e+19), 0)

    def test_negv_kve(self):
        assert_equal(special.kve(3.0, 2.2), special.kve(-3.0, 2.2))

    def test_kve(self):
        kve1 = special.kve(0, 0.2)
        kv1 = special.kv(0, 0.2) * exp(0.2)
        assert_almost_equal(kve1, kv1, 8)
        z = 0.2 + 1j
        kve2 = special.kve(0, z)
        kv2 = special.kv(0, z) * exp(z)
        assert_almost_equal(kve2, kv2, 8)

    def test_kvp_v0n1(self):
        z = 2.2
        assert_almost_equal(-special.kv(1, z), special.kvp(0, z, n=1), 10)

    def test_kvp_n1(self):
        v = 3.0
        z = 2.2
        xc = -special.kv(v + 1, z) + v / z * special.kv(v, z)
        x = special.kvp(v, z, n=1)
        assert_almost_equal(xc, x, 10)

    def test_kvp_n2(self):
        v = 3.0
        z = 2.2
        xc = (z ** 2 + v ** 2 - v) / z ** 2 * special.kv(v, z) + special.kv(v + 1, z) / z
        x = special.kvp(v, z, n=2)
        assert_almost_equal(xc, x, 10)

    def test_y0(self):
        oz = special.y0(0.1)
        ozr = special.yn(0, 0.1)
        assert_almost_equal(oz, ozr, 8)

    def test_y1(self):
        o1 = special.y1(0.1)
        o1r = special.yn(1, 0.1)
        assert_almost_equal(o1, o1r, 8)

    def test_y0_zeros(self):
        yo, ypo = special.y0_zeros(2)
        zo, zpo = special.y0_zeros(2, complex=1)
        all = r_[yo, zo]
        allval = r_[ypo, zpo]
        assert_array_almost_equal(abs(special.yv(0.0, all)), 0.0, 11)
        assert_array_almost_equal(abs(special.yv(1, all) - allval), 0.0, 11)

    def test_y1_zeros(self):
        y1 = special.y1_zeros(1)
        assert_array_almost_equal(y1, (array([2.19714]), array([0.52079])), 5)

    def test_y1p_zeros(self):
        y1p = special.y1p_zeros(1, complex=1)
        assert_array_almost_equal(y1p, (array([0.5768 + 0.904j]), array([-0.7635 + 0.5892j])), 3)

    def test_yn_zeros(self):
        an = special.yn_zeros(4, 2)
        assert_array_almost_equal(an, array([5.64515, 9.36162]), 5)
        an = special.yn_zeros(443, 5)
        assert_allclose(an, [450.1357309157809, 463.05692376675, 472.80651546418665, 481.27353184725627, 488.98055964441374], rtol=1e-15)

    def test_ynp_zeros(self):
        ao = special.ynp_zeros(0, 2)
        assert_array_almost_equal(ao, array([2.19714133, 5.42968104]), 6)
        ao = special.ynp_zeros(43, 5)
        assert_allclose(special.yvp(43, ao), 0, atol=1e-15)
        ao = special.ynp_zeros(443, 5)
        assert_allclose(special.yvp(443, ao), 0, atol=1e-09)

    def test_ynp_zeros_large_order(self):
        ao = special.ynp_zeros(443, 5)
        assert_allclose(special.yvp(443, ao), 0, atol=1e-14)

    def test_yn(self):
        yn2n = special.yn(1, 0.2)
        assert_almost_equal(yn2n, -3.323824988111847, 8)

    def test_negv_yv(self):
        assert_almost_equal(special.yv(-3, 2), -special.yv(3, 2), 14)

    def test_yv(self):
        yv2 = special.yv(1, 0.2)
        assert_almost_equal(yv2, -3.323824988111847, 8)

    def test_negv_yve(self):
        assert_almost_equal(special.yve(-3, 2), -special.yve(3, 2), 14)

    def test_yve(self):
        yve2 = special.yve(1, 0.2)
        assert_almost_equal(yve2, -3.323824988111847, 8)
        yve2r = special.yv(1, 0.2 + 1j) * exp(-1)
        yve22 = special.yve(1, 0.2 + 1j)
        assert_almost_equal(yve22, yve2r, 8)

    def test_yvp(self):
        yvpr = (special.yv(1, 0.2) - special.yv(3, 0.2)) / 2.0
        yvp1 = special.yvp(2, 0.2)
        assert_array_almost_equal(yvp1, yvpr, 10)

    def _cephes_vs_amos_points(self):
        """Yield points at which to compare Cephes implementation to AMOS"""
        v = [-120, -100.3, -20.0, -10.0, -1.0, -0.5, 0.0, 1.0, 12.49, 120.0, 301]
        z = [-1300, -11, -10, -1, 1.0, 10.0, 200.5, 401.0, 600.5, 700.6, 1300, 10003]
        yield from itertools.product(v, z)
        yield from itertools.product(0.5 + arange(-60, 60), [3.5])

    def check_cephes_vs_amos(self, f1, f2, rtol=1e-11, atol=0, skip=None):
        for v, z in self._cephes_vs_amos_points():
            if skip is not None and skip(v, z):
                continue
            c1, c2, c3 = (f1(v, z), f1(v, z + 0j), f2(int(v), z))
            if np.isinf(c1):
                assert_(np.abs(c2) >= 1e+300, (v, z))
            elif np.isnan(c1):
                assert_(c2.imag != 0, (v, z))
            else:
                assert_allclose(c1, c2, err_msg=(v, z), rtol=rtol, atol=atol)
                if v == int(v):
                    assert_allclose(c3, c2, err_msg=(v, z), rtol=rtol, atol=atol)

    @pytest.mark.xfail(platform.machine() == 'ppc64le', reason='fails on ppc64le')
    def test_jv_cephes_vs_amos(self):
        self.check_cephes_vs_amos(special.jv, special.jn, rtol=1e-10, atol=1e-305)

    @pytest.mark.xfail(platform.machine() == 'ppc64le', reason='fails on ppc64le')
    def test_yv_cephes_vs_amos(self):
        self.check_cephes_vs_amos(special.yv, special.yn, rtol=1e-11, atol=1e-305)

    def test_yv_cephes_vs_amos_only_small_orders(self):

        def skipper(v, z):
            return abs(v) > 50
        self.check_cephes_vs_amos(special.yv, special.yn, rtol=1e-11, atol=1e-305, skip=skipper)

    def test_iv_cephes_vs_amos(self):
        with np.errstate(all='ignore'):
            self.check_cephes_vs_amos(special.iv, special.iv, rtol=5e-09, atol=1e-305)

    @pytest.mark.slow
    def test_iv_cephes_vs_amos_mass_test(self):
        N = 1000000
        np.random.seed(1)
        v = np.random.pareto(0.5, N) * (-1) ** np.random.randint(2, size=N)
        x = np.random.pareto(0.2, N) * (-1) ** np.random.randint(2, size=N)
        imsk = np.random.randint(8, size=N) == 0
        v[imsk] = v[imsk].astype(int)
        with np.errstate(all='ignore'):
            c1 = special.iv(v, x)
            c2 = special.iv(v, x + 0j)
            c1[abs(c1) > 1e+300] = np.inf
            c2[abs(c2) > 1e+300] = np.inf
            c1[abs(c1) < 1e-300] = 0
            c2[abs(c2) < 1e-300] = 0
            dc = abs(c1 / c2 - 1)
            dc[np.isnan(dc)] = 0
        k = np.argmax(dc)
        assert_(dc[k] < 2e-07, (v[k], x[k], special.iv(v[k], x[k]), special.iv(v[k], x[k] + 0j)))

    def test_kv_cephes_vs_amos(self):
        self.check_cephes_vs_amos(special.kv, special.kn, rtol=1e-09, atol=1e-305)
        self.check_cephes_vs_amos(special.kv, special.kv, rtol=1e-09, atol=1e-305)

    def test_ticket_623(self):
        assert_allclose(special.jv(3, 4), 0.43017147387562193)
        assert_allclose(special.jv(301, 1300), 0.0183487151115275)
        assert_allclose(special.jv(301, 1296.0682), -0.0224174325312048)

    def test_ticket_853(self):
        """Negative-order Bessels"""
        assert_allclose(special.jv(-1, 1), -0.4400505857449335)
        assert_allclose(special.jv(-2, 1), 0.1149034849319005)
        assert_allclose(special.yv(-1, 1), 0.7812128213002887)
        assert_allclose(special.yv(-2, 1), -1.650682606816255)
        assert_allclose(special.iv(-1, 1), 0.5651591039924851)
        assert_allclose(special.iv(-2, 1), 0.1357476697670383)
        assert_allclose(special.kv(-1, 1), 0.6019072301972347)
        assert_allclose(special.kv(-2, 1), 1.624838898635178)
        assert_allclose(special.jv(-0.5, 1), 0.4310988680183761)
        assert_allclose(special.yv(-0.5, 1), 0.6713967071418031)
        assert_allclose(special.iv(-0.5, 1), 1.231200214592967)
        assert_allclose(special.kv(-0.5, 1), 0.4610685044478945)
        assert_allclose(special.jv(-1, 1 + 0j), -0.4400505857449335)
        assert_allclose(special.jv(-2, 1 + 0j), 0.1149034849319005)
        assert_allclose(special.yv(-1, 1 + 0j), 0.7812128213002887)
        assert_allclose(special.yv(-2, 1 + 0j), -1.650682606816255)
        assert_allclose(special.iv(-1, 1 + 0j), 0.5651591039924851)
        assert_allclose(special.iv(-2, 1 + 0j), 0.1357476697670383)
        assert_allclose(special.kv(-1, 1 + 0j), 0.6019072301972347)
        assert_allclose(special.kv(-2, 1 + 0j), 1.624838898635178)
        assert_allclose(special.jv(-0.5, 1 + 0j), 0.4310988680183761)
        assert_allclose(special.jv(-0.5, 1 + 1j), 0.2628946385649065 - 0.827050182040562j)
        assert_allclose(special.yv(-0.5, 1 + 0j), 0.6713967071418031)
        assert_allclose(special.yv(-0.5, 1 + 1j), 0.967901282890131 + 0.0602046062142816j)
        assert_allclose(special.iv(-0.5, 1 + 0j), 1.231200214592967)
        assert_allclose(special.iv(-0.5, 1 + 1j), 0.77070737376928 + 0.39891821043561j)
        assert_allclose(special.kv(-0.5, 1 + 0j), 0.4610685044478945)
        assert_allclose(special.kv(-0.5, 1 + 1j), 0.06868578341999 - 0.38157825981268j)
        assert_allclose(special.jve(-0.5, 1 + 0.3j), special.jv(-0.5, 1 + 0.3j) * exp(-0.3))
        assert_allclose(special.yve(-0.5, 1 + 0.3j), special.yv(-0.5, 1 + 0.3j) * exp(-0.3))
        assert_allclose(special.ive(-0.5, 0.3 + 1j), special.iv(-0.5, 0.3 + 1j) * exp(-0.3))
        assert_allclose(special.kve(-0.5, 0.3 + 1j), special.kv(-0.5, 0.3 + 1j) * exp(0.3 + 1j))
        assert_allclose(special.hankel1(-0.5, 1 + 1j), special.jv(-0.5, 1 + 1j) + 1j * special.yv(-0.5, 1 + 1j))
        assert_allclose(special.hankel2(-0.5, 1 + 1j), special.jv(-0.5, 1 + 1j) - 1j * special.yv(-0.5, 1 + 1j))

    def test_ticket_854(self):
        """Real-valued Bessel domains"""
        assert_(isnan(special.jv(0.5, -1)))
        assert_(isnan(special.iv(0.5, -1)))
        assert_(isnan(special.yv(0.5, -1)))
        assert_(isnan(special.yv(1, -1)))
        assert_(isnan(special.kv(0.5, -1)))
        assert_(isnan(special.kv(1, -1)))
        assert_(isnan(special.jve(0.5, -1)))
        assert_(isnan(special.ive(0.5, -1)))
        assert_(isnan(special.yve(0.5, -1)))
        assert_(isnan(special.yve(1, -1)))
        assert_(isnan(special.kve(0.5, -1)))
        assert_(isnan(special.kve(1, -1)))
        assert_(isnan(special.airye(-1)[0:2]).all(), special.airye(-1))
        assert_(not isnan(special.airye(-1)[2:4]).any(), special.airye(-1))

    def test_gh_7909(self):
        assert_(special.kv(1.5, 0) == np.inf)
        assert_(special.kve(1.5, 0) == np.inf)

    def test_ticket_503(self):
        """Real-valued Bessel I overflow"""
        assert_allclose(special.iv(1, 700), 1.528500390233901e+302)
        assert_allclose(special.iv(1000, 1120), 1.301564549405821e+301)

    def test_iv_hyperg_poles(self):
        assert_allclose(special.iv(-0.5, 1), 1.231200214592967)

    def iv_series(self, v, z, n=200):
        k = arange(0, n).astype(double)
        r = (v + 2 * k) * log(0.5 * z) - special.gammaln(k + 1) - special.gammaln(v + k + 1)
        r[isnan(r)] = inf
        r = exp(r)
        err = abs(r).max() * finfo(double).eps * n + abs(r[-1]) * 10
        return (r.sum(), err)

    def test_i0_series(self):
        for z in [1.0, 10.0, 200.5]:
            value, err = self.iv_series(0, z)
            assert_allclose(special.i0(z), value, atol=err, err_msg=z)

    def test_i1_series(self):
        for z in [1.0, 10.0, 200.5]:
            value, err = self.iv_series(1, z)
            assert_allclose(special.i1(z), value, atol=err, err_msg=z)

    def test_iv_series(self):
        for v in [-20.0, -10.0, -1.0, 0.0, 1.0, 12.49, 120.0]:
            for z in [1.0, 10.0, 200.5, -1 + 2j]:
                value, err = self.iv_series(v, z)
                assert_allclose(special.iv(v, z), value, atol=err, err_msg=(v, z))

    def test_i0(self):
        values = [[0.0, 1.0], [1e-10, 1.0], [0.1, 0.9071009258], [0.5, 0.6450352706], [1.0, 0.4657596077], [2.5, 0.2700464416], [5.0, 0.1835408126], [20.0, 0.0897803119]]
        for i, (x, v) in enumerate(values):
            cv = special.i0(x) * exp(-x)
            assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)

    def test_i0e(self):
        oize = special.i0e(0.1)
        oizer = special.ive(0, 0.1)
        assert_almost_equal(oize, oizer, 8)

    def test_i1(self):
        values = [[0.0, 0.0], [1e-10, 4.9999999995e-11], [0.1, 0.0452984468], [0.5, 0.1564208032], [1.0, 0.2079104154], [5.0, 0.1639722669], [20.0, 0.0875062222]]
        for i, (x, v) in enumerate(values):
            cv = special.i1(x) * exp(-x)
            assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)

    def test_i1e(self):
        oi1e = special.i1e(0.1)
        oi1er = special.ive(1, 0.1)
        assert_almost_equal(oi1e, oi1er, 8)

    def test_iti0k0(self):
        iti0 = array(special.iti0k0(5))
        assert_array_almost_equal(iti0, array([31.8486677761698, 1.5673873907283657]), 5)

    def test_it2i0k0(self):
        it2k = special.it2i0k0(0.1)
        assert_array_almost_equal(it2k, array([0.0012503906973464409, 3.3309450354686687]), 6)

    def test_iv(self):
        iv1 = special.iv(0, 0.1) * exp(-0.1)
        assert_almost_equal(iv1, 0.9071009257823011, 10)

    def test_negv_ive(self):
        assert_equal(special.ive(3, 2), special.ive(-3, 2))

    def test_ive(self):
        ive1 = special.ive(0, 0.1)
        iv1 = special.iv(0, 0.1) * exp(-0.1)
        assert_almost_equal(ive1, iv1, 10)

    def test_ivp0(self):
        assert_almost_equal(special.iv(1, 2), special.ivp(0, 2), 10)

    def test_ivp(self):
        y = (special.iv(0, 2) + special.iv(2, 2)) / 2
        x = special.ivp(1, 2)
        assert_almost_equal(x, y, 10)