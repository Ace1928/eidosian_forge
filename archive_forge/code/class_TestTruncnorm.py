import warnings
import re
import sys
import pickle
from pathlib import Path
import os
import json
import platform
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import numpy
import numpy as np
from numpy import typecodes, array
from numpy.lib.recfunctions import rec_append_fields
from scipy import special
from scipy._lib._util import check_random_state
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce
import scipy.stats.distributions
from scipy.special import xlogy, polygamma, entr
from scipy.stats._distr_params import distcont, invdistcont
from .test_discrete_basic import distdiscrete, invdistdiscrete
from scipy.stats._continuous_distns import FitDataError, _argus_phi
from scipy.optimize import root, fmin, differential_evolution
from itertools import product
class TestTruncnorm:

    def setup_method(self):
        np.random.seed(1234)

    @pytest.mark.parametrize('a, b, ref', [(0, 100, 0.7257913526447274), (0.6, 0.7, -2.3027610681852573), (1e-06, 2e-06, -13.815510557964274)])
    def test_entropy(self, a, b, ref):
        assert_allclose(stats.truncnorm.entropy(a, b), ref, rtol=1e-10)

    @pytest.mark.parametrize('a, b, ref', [(1e-11, 10000000000.0, 0.725791352640738), (1e-100, 1e+100, 0.7257913526447274), (-1e-100, 1e+100, 0.7257913526447274), (-1e+100, 1e+100, 1.4189385332046727)])
    def test_extreme_entropy(self, a, b, ref):
        assert_allclose(stats.truncnorm.entropy(a, b), ref, rtol=1e-14)

    def test_ppf_ticket1131(self):
        vals = stats.truncnorm.ppf([-0.5, 0, 0.0001, 0.5, 1 - 0.0001, 1, 2], -1.0, 1.0, loc=[3] * 7, scale=2)
        expected = np.array([np.nan, 1, 1.00056419, 3, 4.99943581, 5, np.nan])
        assert_array_almost_equal(vals, expected)

    def test_isf_ticket1131(self):
        vals = stats.truncnorm.isf([-0.5, 0, 0.0001, 0.5, 1 - 0.0001, 1, 2], -1.0, 1.0, loc=[3] * 7, scale=2)
        expected = np.array([np.nan, 5, 4.99943581, 3, 1.00056419, 1, np.nan])
        assert_array_almost_equal(vals, expected)

    def test_gh_2477_small_values(self):
        low, high = (-11, -10)
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)
        low, high = (10, 11)
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)

    def test_gh_2477_large_values(self):
        low, high = (100, 101)
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        (assert_(low <= x.min() <= x.max() <= high), str([low, high, x]))
        low, high = (1000, 1001)
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)
        low, high = (10000, 10001)
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)
        low, high = (-10001, -10000)
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)

    def test_gh_9403_nontail_values(self):
        for low, high in [[3, 4], [-4, -3]]:
            xvals = np.array([-np.inf, low, high, np.inf])
            xmid = (high + low) / 2.0
            cdfs = stats.truncnorm.cdf(xvals, low, high)
            sfs = stats.truncnorm.sf(xvals, low, high)
            pdfs = stats.truncnorm.pdf(xvals, low, high)
            expected_cdfs = np.array([0, 0, 1, 1])
            expected_sfs = np.array([1.0, 1.0, 0.0, 0.0])
            expected_pdfs = np.array([0, 3.3619772, 0.1015229, 0])
            if low < 0:
                expected_pdfs = np.array([0, 0.1015229, 3.3619772, 0])
            assert_almost_equal(cdfs, expected_cdfs)
            assert_almost_equal(sfs, expected_sfs)
            assert_almost_equal(pdfs, expected_pdfs)
            assert_almost_equal(np.log(expected_pdfs[1] / expected_pdfs[2]), low + 0.5)
            pvals = np.array([0, 0.5, 1.0])
            ppfs = stats.truncnorm.ppf(pvals, low, high)
            expected_ppfs = np.array([low, np.sign(low) * 3.1984741, high])
            assert_almost_equal(ppfs, expected_ppfs)
            if low < 0:
                assert_almost_equal(stats.truncnorm.sf(xmid, low, high), 0.8475544278436675)
                assert_almost_equal(stats.truncnorm.cdf(xmid, low, high), 0.1524455721563326)
            else:
                assert_almost_equal(stats.truncnorm.cdf(xmid, low, high), 0.8475544278436675)
                assert_almost_equal(stats.truncnorm.sf(xmid, low, high), 0.1524455721563326)
            pdf = stats.truncnorm.pdf(xmid, low, high)
            assert_almost_equal(np.log(pdf / expected_pdfs[2]), (xmid + 0.25) / 2)

    def test_gh_9403_medium_tail_values(self):
        for low, high in [[39, 40], [-40, -39]]:
            xvals = np.array([-np.inf, low, high, np.inf])
            xmid = (high + low) / 2.0
            cdfs = stats.truncnorm.cdf(xvals, low, high)
            sfs = stats.truncnorm.sf(xvals, low, high)
            pdfs = stats.truncnorm.pdf(xvals, low, high)
            expected_cdfs = np.array([0, 0, 1, 1])
            expected_sfs = np.array([1.0, 1.0, 0.0, 0.0])
            expected_pdfs = np.array([0, 39.0256074, 2.73349092e-16, 0])
            if low < 0:
                expected_pdfs = np.array([0, 2.73349092e-16, 39.0256074, 0])
            assert_almost_equal(cdfs, expected_cdfs)
            assert_almost_equal(sfs, expected_sfs)
            assert_almost_equal(pdfs, expected_pdfs)
            assert_almost_equal(np.log(expected_pdfs[1] / expected_pdfs[2]), low + 0.5)
            pvals = np.array([0, 0.5, 1.0])
            ppfs = stats.truncnorm.ppf(pvals, low, high)
            expected_ppfs = np.array([low, np.sign(low) * 39.01775731, high])
            assert_almost_equal(ppfs, expected_ppfs)
            cdfs = stats.truncnorm.cdf(ppfs, low, high)
            assert_almost_equal(cdfs, pvals)
            if low < 0:
                assert_almost_equal(stats.truncnorm.sf(xmid, low, high), 0.9999999970389126)
                assert_almost_equal(stats.truncnorm.cdf(xmid, low, high), 2.961048103554866e-09)
            else:
                assert_almost_equal(stats.truncnorm.cdf(xmid, low, high), 0.9999999970389126)
                assert_almost_equal(stats.truncnorm.sf(xmid, low, high), 2.961048103554866e-09)
            pdf = stats.truncnorm.pdf(xmid, low, high)
            assert_almost_equal(np.log(pdf / expected_pdfs[2]), (xmid + 0.25) / 2)
            xvals = np.linspace(low, high, 11)
            xvals2 = -xvals[::-1]
            assert_almost_equal(stats.truncnorm.cdf(xvals, low, high), stats.truncnorm.sf(xvals2, -high, -low)[::-1])
            assert_almost_equal(stats.truncnorm.sf(xvals, low, high), stats.truncnorm.cdf(xvals2, -high, -low)[::-1])
            assert_almost_equal(stats.truncnorm.pdf(xvals, low, high), stats.truncnorm.pdf(xvals2, -high, -low)[::-1])

    def test_cdf_tail_15110_14753(self):
        assert_allclose(stats.truncnorm(13.0, 15.0).cdf(14.0), 0.9999987259565642)
        assert_allclose(stats.truncnorm(8, np.inf).cdf(8.3), 0.916322090732754)
    _truncnorm_stats_data = [[-30, 30, 0.0, 1.0, 0.0, 0.0], [-10, 10, 0.0, 1.0, 0.0, -1.4927521335810455e-19], [-3, 3, 0.0, 0.9733369246625415, 0.0, -0.17111443639774404], [-2, 2, 0.0, 0.7737413035499232, 0.0, -0.6344632828703505], [0, np.inf, 0.7978845608028654, 0.3633802276324187, 0.995271746431156, 0.8691773036059741], [-np.inf, 0, -0.7978845608028654, 0.3633802276324187, -0.995271746431156, 0.8691773036059741], [-1, 3, 0.282786110727154, 0.6161417353578293, 0.5393018494027877, -0.20582065135274694], [-3, 1, -0.282786110727154, 0.6161417353578293, -0.5393018494027877, -0.20582065135274694], [-10, -9, -9.108456288012409, 0.011448805821636248, -1.8985607290949496, 5.0733461105025075]]
    _truncnorm_stats_data = np.array(_truncnorm_stats_data)

    @pytest.mark.parametrize('case', _truncnorm_stats_data)
    def test_moments(self, case):
        a, b, m0, v0, s0, k0 = case
        m, v, s, k = stats.truncnorm.stats(a, b, moments='mvsk')
        assert_allclose([m, v, s, k], [m0, v0, s0, k0], atol=1e-17)

    def test_9902_moments(self):
        m, v = stats.truncnorm.stats(0, np.inf, moments='mv')
        assert_almost_equal(m, 0.79788456)
        assert_almost_equal(v, 0.36338023)

    def test_gh_1489_trac_962_rvs(self):
        low, high = (10, 15)
        x = stats.truncnorm.rvs(low, high, 0, 1, size=10)
        assert_(low < x.min() < x.max() < high)

    def test_gh_11299_rvs(self):
        low = [-10, 10, -np.inf, -5, -np.inf, -np.inf, -45, -45, 40, -10, 40]
        high = [-5, 11, 5, np.inf, 40, -40, 40, -40, 45, np.inf, np.inf]
        x = stats.truncnorm.rvs(low, high, size=(5, len(low)))
        assert np.shape(x) == (5, len(low))
        assert_(np.all(low <= x.min(axis=0)))
        assert_(np.all(x.max(axis=0) <= high))

    def test_rvs_Generator(self):
        if hasattr(np.random, 'default_rng'):
            stats.truncnorm.rvs(-10, -5, size=5, random_state=np.random.default_rng())

    def test_logcdf_gh17064(self):
        a = np.array([-np.inf, -np.inf, -8, -np.inf, 10])
        b = np.array([np.inf, np.inf, 8, 10, np.inf])
        x = np.array([10, 7.5, 7.5, 9, 20])
        expected = [-7.619853024160525e-24, -3.190891672910947e-14, -3.128682067168231e-14, -1.1285122074235991e-19, -3.61374964828753e-66]
        assert_allclose(stats.truncnorm(a, b).logcdf(x), expected)
        assert_allclose(stats.truncnorm(-b, -a).logsf(-x), expected)

    def test_moments_gh18634(self):
        res = stats.truncnorm(-2, 3).moment(5)
        ref = 1.645309620208361
        assert_allclose(res, ref)