import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
@pytest.mark.slow
@check_version(mpmath, '0.19')
class TestCDFlib:

    @pytest.mark.xfail(run=False)
    def test_bdtrik(self):
        _assert_inverts(sp.bdtrik, _binomial_cdf, 0, [ProbArg(), IntArg(1, 1000), ProbArg()], rtol=0.0001)

    def test_bdtrin(self):
        _assert_inverts(sp.bdtrin, _binomial_cdf, 1, [IntArg(1, 1000), ProbArg(), ProbArg()], rtol=0.0001, endpt_atol=[None, None, 1e-06])

    def test_btdtria(self):
        _assert_inverts(sp.btdtria, lambda a, b, x: mpmath.betainc(a, b, x2=x, regularized=True), 0, [ProbArg(), Arg(0, 100.0, inclusive_a=False), Arg(0, 1, inclusive_a=False, inclusive_b=False)], rtol=1e-06)

    def test_btdtrib(self):
        _assert_inverts(sp.btdtrib, lambda a, b, x: mpmath.betainc(a, b, x2=x, regularized=True), 1, [Arg(0, 100.0, inclusive_a=False), ProbArg(), Arg(0, 1, inclusive_a=False, inclusive_b=False)], rtol=1e-07, endpt_atol=[None, 1e-18, 1e-15])

    @pytest.mark.xfail(run=False)
    def test_fdtridfd(self):
        _assert_inverts(sp.fdtridfd, _f_cdf, 1, [IntArg(1, 100), ProbArg(), Arg(0, 100, inclusive_a=False)], rtol=1e-07)

    def test_gdtria(self):
        _assert_inverts(sp.gdtria, lambda a, b, x: mpmath.gammainc(b, b=a * x, regularized=True), 0, [ProbArg(), Arg(0, 1000.0, inclusive_a=False), Arg(0, 10000.0, inclusive_a=False)], rtol=1e-07, endpt_atol=[None, 1e-07, 1e-10])

    def test_gdtrib(self):
        _assert_inverts(sp.gdtrib, lambda a, b, x: mpmath.gammainc(b, b=a * x, regularized=True), 1, [Arg(0, 100.0, inclusive_a=False), ProbArg(), Arg(0, 1000.0, inclusive_a=False)], rtol=1e-05)

    def test_gdtrix(self):
        _assert_inverts(sp.gdtrix, lambda a, b, x: mpmath.gammainc(b, b=a * x, regularized=True), 2, [Arg(0, 1000.0, inclusive_a=False), Arg(0, 1000.0, inclusive_a=False), ProbArg()], rtol=1e-07, endpt_atol=[None, 1e-07, 1e-10])

    def test_stdtr(self):
        assert_mpmath_equal(sp.stdtr, _student_t_cdf, [IntArg(1, 100), Arg(1e-10, np.inf)], rtol=1e-07)

    @pytest.mark.xfail(run=False)
    def test_stdtridf(self):
        _assert_inverts(sp.stdtridf, _student_t_cdf, 0, [ProbArg(), Arg()], rtol=1e-07)

    def test_stdtrit(self):
        _assert_inverts(sp.stdtrit, _student_t_cdf, 1, [IntArg(1, 100), ProbArg()], rtol=1e-07, endpt_atol=[None, 1e-10])

    def test_chdtriv(self):
        _assert_inverts(sp.chdtriv, lambda v, x: mpmath.gammainc(v / 2, b=x / 2, regularized=True), 0, [ProbArg(), IntArg(1, 100)], rtol=0.0001)

    @pytest.mark.xfail(run=False)
    def test_chndtridf(self):
        _assert_inverts(sp.chndtridf, _noncentral_chi_cdf, 1, [Arg(0, 100, inclusive_a=False), ProbArg(), Arg(0, 100, inclusive_a=False)], n=1000, rtol=0.0001, atol=1e-15)

    @pytest.mark.xfail(run=False)
    def test_chndtrinc(self):
        _assert_inverts(sp.chndtrinc, _noncentral_chi_cdf, 2, [Arg(0, 100, inclusive_a=False), IntArg(1, 100), ProbArg()], n=1000, rtol=0.0001, atol=1e-15)

    def test_chndtrix(self):
        _assert_inverts(sp.chndtrix, _noncentral_chi_cdf, 0, [ProbArg(), IntArg(1, 100), Arg(0, 100, inclusive_a=False)], n=1000, rtol=0.0001, atol=1e-15, endpt_atol=[1e-06, None, None])

    def test_tklmbda_zero_shape(self):
        one = mpmath.mpf(1)
        assert_mpmath_equal(lambda x: sp.tklmbda(x, 0), lambda x: one / (mpmath.exp(-x) + one), [Arg()], rtol=1e-07)

    def test_tklmbda_neg_shape(self):
        _assert_inverts(sp.tklmbda, _tukey_lmbda_quantile, 0, [ProbArg(), Arg(-25, 0, inclusive_b=False)], spfunc_first=False, rtol=1e-05, endpt_atol=[1e-09, 1e-05])

    @pytest.mark.xfail(run=False)
    def test_tklmbda_pos_shape(self):
        _assert_inverts(sp.tklmbda, _tukey_lmbda_quantile, 0, [ProbArg(), Arg(0, 100, inclusive_a=False)], spfunc_first=False, rtol=1e-05)

    @pytest.mark.parametrize('lmbda', [0.5, 1.0, 8.0])
    def test_tklmbda_lmbda1(self, lmbda):
        bound = 1 / lmbda
        assert_equal(sp.tklmbda([-bound, bound], lmbda), [0.0, 1.0])