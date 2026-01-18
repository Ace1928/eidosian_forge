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
class TestWeibull:

    def test_logpdf(self):
        y = stats.weibull_min.logpdf(0, 1)
        assert_equal(y, 0)

    def test_with_maxima_distrib(self):
        x = 1.5
        a = 2.0
        b = 3.0
        p = stats.weibull_min.pdf(x, a, scale=b)
        assert_allclose(p, np.exp(-0.25) / 3)
        lp = stats.weibull_min.logpdf(x, a, scale=b)
        assert_allclose(lp, -0.25 - np.log(3))
        c = stats.weibull_min.cdf(x, a, scale=b)
        assert_allclose(c, -special.expm1(-0.25))
        lc = stats.weibull_min.logcdf(x, a, scale=b)
        assert_allclose(lc, np.log(-special.expm1(-0.25)))
        s = stats.weibull_min.sf(x, a, scale=b)
        assert_allclose(s, np.exp(-0.25))
        ls = stats.weibull_min.logsf(x, a, scale=b)
        assert_allclose(ls, -0.25)
        s = stats.weibull_min.sf(30, 2, scale=3)
        assert_allclose(s, np.exp(-100))
        ls = stats.weibull_min.logsf(30, 2, scale=3)
        assert_allclose(ls, -100)
        x = -1.5
        p = stats.weibull_max.pdf(x, a, scale=b)
        assert_allclose(p, np.exp(-0.25) / 3)
        lp = stats.weibull_max.logpdf(x, a, scale=b)
        assert_allclose(lp, -0.25 - np.log(3))
        c = stats.weibull_max.cdf(x, a, scale=b)
        assert_allclose(c, np.exp(-0.25))
        lc = stats.weibull_max.logcdf(x, a, scale=b)
        assert_allclose(lc, -0.25)
        s = stats.weibull_max.sf(x, a, scale=b)
        assert_allclose(s, -special.expm1(-0.25))
        ls = stats.weibull_max.logsf(x, a, scale=b)
        assert_allclose(ls, np.log(-special.expm1(-0.25)))
        s = stats.weibull_max.sf(-1e-09, 2, scale=3)
        assert_allclose(s, -special.expm1(-1 / 9000000000000000000))
        ls = stats.weibull_max.logsf(-1e-09, 2, scale=3)
        assert_allclose(ls, np.log(-special.expm1(-1 / 9000000000000000000)))

    @pytest.mark.parametrize('scale', [1.0, 0.1])
    def test_delta_cdf(self, scale):
        delta = stats.weibull_min._delta_cdf(scale * 7.5, scale * 8, 3, scale=scale)
        assert_allclose(delta, 6.053624060118734e-184)

    def test_fit_min(self):
        rng = np.random.default_rng(5985959307161735394)
        c, loc, scale = (2, 3.5, 0.5)
        dist = stats.weibull_min(c, loc, scale)
        rvs = dist.rvs(size=100, random_state=rng)
        c2, loc2, scale2 = stats.weibull_min.fit(rvs, 1.5, floc=3)
        c3, loc3, scale3 = stats.weibull_min.fit(rvs, 1.6, floc=3)
        assert loc2 == loc3 == 3
        assert c2 != c3
        c4, loc4, scale4 = stats.weibull_min.fit(rvs, 3, fscale=3, method='mm')
        assert scale4 == 3
        dist4 = stats.weibull_min(c4, loc4, scale4)
        res = dist4.stats(moments='ms')
        ref = (np.mean(rvs), stats.skew(rvs))
        assert_allclose(res, ref)

    @pytest.mark.parametrize('x, c, ref', [(50, 1, 1.9287498479639178e-22), (1000, 0.8, 8.131269637872743e-110)])
    def test_sf_isf(self, x, c, ref):
        assert_allclose(stats.weibull_min.sf(x, c), ref, rtol=5e-14)
        assert_allclose(stats.weibull_min.isf(ref, c), x, rtol=5e-14)