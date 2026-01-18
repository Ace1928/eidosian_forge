import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
class TestZiNBP:

    def test_pmf_p2(self):
        n, p = zinegbin.convert_params(30, 0.1, 2)
        nb_pmf = nbinom.pmf(100, n, p)
        tnb_pmf = zinegbin.pmf(100, 30, 0.1, 2, 0.01)
        assert_allclose(nb_pmf, tnb_pmf, rtol=1e-05, atol=1e-05)

    def test_logpmf_p2(self):
        n, p = zinegbin.convert_params(10, 1, 2)
        nb_logpmf = nbinom.logpmf(200, n, p)
        tnb_logpmf = zinegbin.logpmf(200, 10, 1, 2, 0.01)
        assert_allclose(nb_logpmf, tnb_logpmf, rtol=0.01, atol=0.01)

    def test_cdf_p2(self):
        n, p = zinegbin.convert_params(30, 0.1, 2)
        nbinom_cdf = nbinom.cdf(10, n, p)
        zinbinom_cdf = zinegbin.cdf(10, 30, 0.1, 2, 0)
        assert_allclose(nbinom_cdf, zinbinom_cdf, rtol=1e-12, atol=1e-12)

    def test_ppf_p2(self):
        n, p = zinegbin.convert_params(100, 1, 2)
        nbinom_ppf = nbinom.ppf(0.27, n, p)
        zinbinom_ppf = zinegbin.ppf(0.27, 100, 1, 2, 0)
        assert_allclose(nbinom_ppf, zinbinom_ppf, rtol=1e-12, atol=1e-12)

    def test_mran_var_p2(self):
        n, p = zinegbin.convert_params(7, 1, 2)
        nbinom_mean, nbinom_var = (nbinom.mean(n, p), nbinom.var(n, p))
        zinb_mean = zinegbin.mean(7, 1, 2, 0)
        zinb_var = zinegbin.var(7, 1, 2, 0)
        assert_allclose(nbinom_mean, zinb_mean, rtol=1e-10)
        assert_allclose(nbinom_var, zinb_var, rtol=1e-10)

    def test_moments_p2(self):
        n, p = zinegbin.convert_params(7, 1, 2)
        nb_m1, nb_m2 = (nbinom.moment(1, n, p), nbinom.moment(2, n, p))
        zinb_m0 = zinegbin.moment(0, 7, 1, 2, 0)
        zinb_m1 = zinegbin.moment(1, 7, 1, 2, 0)
        zinb_m2 = zinegbin.moment(2, 7, 1, 2, 0)
        assert_allclose(1, zinb_m0, rtol=1e-10)
        assert_allclose(nb_m1, zinb_m1, rtol=1e-10)
        assert_allclose(nb_m2, zinb_m2, rtol=1e-10)

    def test_pmf(self):
        n, p = zinegbin.convert_params(1, 0.9, 1)
        nb_logpmf = nbinom.pmf(2, n, p)
        tnb_pmf = zinegbin.pmf(2, 1, 0.9, 2, 0.5)
        assert_allclose(nb_logpmf, tnb_pmf * 2, rtol=1e-07)

    def test_logpmf(self):
        n, p = zinegbin.convert_params(5, 1, 1)
        nb_logpmf = nbinom.logpmf(2, n, p)
        tnb_logpmf = zinegbin.logpmf(2, 5, 1, 1, 0.005)
        assert_allclose(nb_logpmf, tnb_logpmf, rtol=0.01, atol=0.01)

    def test_cdf(self):
        n, p = zinegbin.convert_params(1, 0.9, 1)
        nbinom_cdf = nbinom.cdf(2, n, p)
        zinbinom_cdf = zinegbin.cdf(2, 1, 0.9, 2, 0)
        assert_allclose(nbinom_cdf, zinbinom_cdf, rtol=1e-12, atol=1e-12)

    def test_ppf(self):
        n, p = zinegbin.convert_params(5, 1, 1)
        nbinom_ppf = nbinom.ppf(0.71, n, p)
        zinbinom_ppf = zinegbin.ppf(0.71, 5, 1, 1, 0)
        assert_allclose(nbinom_ppf, zinbinom_ppf, rtol=1e-12, atol=1e-12)

    def test_convert(self):
        n, p = zinegbin.convert_params(25, 0.85, 2)
        n_true, p_true = (1.1764705882352942, 0.04494382022471911)
        assert_allclose(n, n_true, rtol=1e-12, atol=1e-12)
        assert_allclose(p, p_true, rtol=1e-12, atol=1e-12)
        n, p = zinegbin.convert_params(7, 0.17, 1)
        n_true, p_true = (41.17647058823529, 0.8547008547008547)
        assert_allclose(n, n_true, rtol=1e-12, atol=1e-12)
        assert_allclose(p, p_true, rtol=1e-12, atol=1e-12)

    def test_mean_var(self):
        for m in [9, np.array([1, 5, 10])]:
            n, p = zinegbin.convert_params(m, 1, 1)
            nbinom_mean, nbinom_var = (nbinom.mean(n, p), nbinom.var(n, p))
            zinb_mean = zinegbin.mean(m, 1, 1, 0)
            zinb_var = zinegbin.var(m, 1, 1, 0)
            assert_allclose(nbinom_mean, zinb_mean, rtol=1e-10)
            assert_allclose(nbinom_var, zinb_var, rtol=1e-10)

    def test_moments(self):
        n, p = zinegbin.convert_params(9, 1, 1)
        nb_m1, nb_m2 = (nbinom.moment(1, n, p), nbinom.moment(2, n, p))
        zinb_m0 = zinegbin.moment(0, 9, 1, 1, 0)
        zinb_m1 = zinegbin.moment(1, 9, 1, 1, 0)
        zinb_m2 = zinegbin.moment(2, 9, 1, 1, 0)
        assert_allclose(1, zinb_m0, rtol=1e-10)
        assert_allclose(nb_m1, zinb_m1, rtol=1e-10)
        assert_allclose(nb_m2, zinb_m2, rtol=1e-10)