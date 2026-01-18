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
class TestExpect:

    def test_norm(self):
        v = stats.norm.expect(lambda x: (x - 5) * (x - 5), loc=5, scale=2)
        assert_almost_equal(v, 4, decimal=14)
        m = stats.norm.expect(lambda x: x, loc=5, scale=2)
        assert_almost_equal(m, 5, decimal=14)
        lb = stats.norm.ppf(0.05, loc=5, scale=2)
        ub = stats.norm.ppf(0.95, loc=5, scale=2)
        prob90 = stats.norm.expect(lambda x: 1, loc=5, scale=2, lb=lb, ub=ub)
        assert_almost_equal(prob90, 0.9, decimal=14)
        prob90c = stats.norm.expect(lambda x: 1, loc=5, scale=2, lb=lb, ub=ub, conditional=True)
        assert_almost_equal(prob90c, 1.0, decimal=14)

    def test_beta(self):
        v = stats.beta.expect(lambda x: (x - 19 / 3.0) * (x - 19 / 3.0), args=(10, 5), loc=5, scale=2)
        assert_almost_equal(v, 1.0 / 18.0, decimal=13)
        m = stats.beta.expect(lambda x: x, args=(10, 5), loc=5.0, scale=2.0)
        assert_almost_equal(m, 19 / 3.0, decimal=13)
        ub = stats.beta.ppf(0.95, 10, 10, loc=5, scale=2)
        lb = stats.beta.ppf(0.05, 10, 10, loc=5, scale=2)
        prob90 = stats.beta.expect(lambda x: 1.0, args=(10, 10), loc=5.0, scale=2.0, lb=lb, ub=ub, conditional=False)
        assert_almost_equal(prob90, 0.9, decimal=13)
        prob90c = stats.beta.expect(lambda x: 1, args=(10, 10), loc=5, scale=2, lb=lb, ub=ub, conditional=True)
        assert_almost_equal(prob90c, 1.0, decimal=13)

    def test_hypergeom(self):
        m_true, v_true = stats.hypergeom.stats(20, 10, 8, loc=5.0)
        m = stats.hypergeom.expect(lambda x: x, args=(20, 10, 8), loc=5.0)
        assert_almost_equal(m, m_true, decimal=13)
        v = stats.hypergeom.expect(lambda x: (x - 9.0) ** 2, args=(20, 10, 8), loc=5.0)
        assert_almost_equal(v, v_true, decimal=14)
        v_bounds = stats.hypergeom.expect(lambda x: (x - 9.0) ** 2, args=(20, 10, 8), loc=5.0, lb=5, ub=13)
        assert_almost_equal(v_bounds, v_true, decimal=14)
        prob_true = 1 - stats.hypergeom.pmf([5, 13], 20, 10, 8, loc=5).sum()
        prob_bounds = stats.hypergeom.expect(lambda x: 1, args=(20, 10, 8), loc=5.0, lb=6, ub=12)
        assert_almost_equal(prob_bounds, prob_true, decimal=13)
        prob_bc = stats.hypergeom.expect(lambda x: 1, args=(20, 10, 8), loc=5.0, lb=6, ub=12, conditional=True)
        assert_almost_equal(prob_bc, 1, decimal=14)
        prob_b = stats.hypergeom.expect(lambda x: 1, args=(20, 10, 8), lb=0, ub=8)
        assert_almost_equal(prob_b, 1, decimal=13)

    def test_poisson(self):
        prob_bounds = stats.poisson.expect(lambda x: 1, args=(2,), lb=3, conditional=False)
        prob_b_true = 1 - stats.poisson.cdf(2, 2)
        assert_almost_equal(prob_bounds, prob_b_true, decimal=14)
        prob_lb = stats.poisson.expect(lambda x: 1, args=(2,), lb=2, conditional=True)
        assert_almost_equal(prob_lb, 1, decimal=14)

    def test_genhalflogistic(self):
        halflog = stats.genhalflogistic
        res1 = halflog.expect(args=(1.5,))
        halflog.expect(args=(0.5,))
        res2 = halflog.expect(args=(1.5,))
        assert_almost_equal(res1, res2, decimal=14)

    def test_rice_overflow(self):
        assert_(np.isfinite(stats.rice.pdf(999, 0.74)))
        assert_(np.isfinite(stats.rice.expect(lambda x: 1, args=(0.74,))))
        assert_(np.isfinite(stats.rice.expect(lambda x: 2, args=(0.74,))))
        assert_(np.isfinite(stats.rice.expect(lambda x: 3, args=(0.74,))))

    def test_logser(self):
        p, loc = (0.3, 3)
        res_0 = stats.logser.expect(lambda k: k, args=(p,))
        assert_allclose(res_0, p / (p - 1.0) / np.log(1.0 - p), atol=1e-15)
        res_l = stats.logser.expect(lambda k: k, args=(p,), loc=loc)
        assert_allclose(res_l, res_0 + loc, atol=1e-15)

    def test_skellam(self):
        p1, p2 = (18, 22)
        m1 = stats.skellam.expect(lambda x: x, args=(p1, p2))
        m2 = stats.skellam.expect(lambda x: x ** 2, args=(p1, p2))
        assert_allclose(m1, p1 - p2, atol=1e-12)
        assert_allclose(m2 - m1 ** 2, p1 + p2, atol=1e-12)

    def test_randint(self):
        lo, hi = (0, 113)
        res = stats.randint.expect(lambda x: x, (lo, hi))
        assert_allclose(res, sum((_ for _ in range(lo, hi))) / (hi - lo), atol=1e-15)

    def test_zipf(self):
        assert_warns(RuntimeWarning, stats.zipf.expect, lambda x: x ** 2, (2,))

    def test_discrete_kwds(self):
        n0 = stats.poisson.expect(lambda x: 1, args=(2,))
        n1 = stats.poisson.expect(lambda x: 1, args=(2,), maxcount=1001, chunksize=32, tolerance=1e-08)
        assert_almost_equal(n0, n1, decimal=14)

    def test_moment(self):

        def poiss_moment5(mu):
            return mu ** 5 + 10 * mu ** 4 + 25 * mu ** 3 + 15 * mu ** 2 + mu
        for mu in [5, 7]:
            m5 = stats.poisson.moment(5, mu)
            assert_allclose(m5, poiss_moment5(mu), rtol=1e-10)

    def test_challenging_cases_gh8928(self):
        assert_allclose(stats.norm.expect(loc=36, scale=1.0), 36)
        assert_allclose(stats.norm.expect(loc=40, scale=1.0), 40)
        assert_allclose(stats.norm.expect(loc=10, scale=0.1), 10)
        assert_allclose(stats.gamma.expect(args=(148,)), 148)
        assert_allclose(stats.logistic.expect(loc=85), 85)

    def test_lb_ub_gh15855(self):
        dist = stats.uniform
        ref = dist.mean(loc=10, scale=5)
        assert_allclose(dist.expect(loc=10, scale=5), ref)
        assert_allclose(dist.expect(loc=10, scale=5, lb=9, ub=16), ref)
        assert_allclose(dist.expect(loc=10, scale=5, lb=11, ub=14), ref * 0.6)
        assert_allclose(dist.expect(loc=10, scale=5, lb=11, ub=14, conditional=True), ref)
        assert_allclose(dist.expect(loc=10, scale=5, lb=11, ub=13), 12 * 0.4)
        assert_allclose(dist.expect(loc=10, scale=5, lb=13, ub=11), -12 * 0.4)
        assert_allclose(dist.expect(loc=10, scale=5, lb=13, ub=11, conditional=True), 12)