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
class TestGenHyperbolic:

    def setup_method(self):
        np.random.seed(1234)

    def test_pdf_r(self):
        vals_R = np.array([2.94895678275316e-13, 1.75746848647696e-10, 9.48149804073045e-08, 4.17862521692026e-05, 0.0103947630463822, 0.240864958986839, 0.162833527161649, 0.0374609592899472, 0.00634894847327781, 0.000941920705790324])
        lmbda, alpha, beta = (2, 2, 1)
        mu, delta = (0.5, 1.5)
        args = (lmbda, alpha * delta, beta * delta)
        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        x = np.linspace(-10, 10, 10)
        assert_allclose(gh.pdf(x), vals_R, atol=0, rtol=1e-13)

    def test_cdf_r(self):
        vals_R = np.array([1.01881590921421e-13, 6.13697274983578e-11, 3.37504977637992e-08, 1.55258698166181e-05, 0.00447005453832497, 0.228935323956347, 0.755759458895243, 0.953061062884484, 0.992598013917513, 0.998942646586662])
        lmbda, alpha, beta = (2, 2, 1)
        mu, delta = (0.5, 1.5)
        args = (lmbda, alpha * delta, beta * delta)
        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        x = np.linspace(-10, 10, 10)
        assert_allclose(gh.cdf(x), vals_R, atol=0, rtol=1e-06)

    @pytest.mark.parametrize('x, p, a, b, loc, scale, ref', [(-15, 2, 3, 1.5, 0.5, 1.5, 4.770036428808252e-20), (-15, 10, 1.5, 0.25, 1, 5, 0.03282964575089294), (-15, 10, 1.5, 1.375, 0, 1, 3.3711159600215594e-23), (-15, 0.125, 1.5, 1.49995, 0, 1, 4.729401428898605e-23), (-1, 0.125, 1.5, 1.49995, 0, 1, 0.0003565725914786859), (5, -0.125, 1.5, 1.49995, 0, 1, 0.2600651974023352), (5, -0.125, 1000, 999, 0, 1, 5.923270556517253e-28), (20, -0.125, 1000, 999, 0, 1, 0.23452293711665634), (40, -0.125, 1000, 999, 0, 1, 0.9999648749561968), (60, -0.125, 1000, 999, 0, 1, 0.9999999999975475)])
    def test_cdf_mpmath(self, x, p, a, b, loc, scale, ref):
        cdf = stats.genhyperbolic.cdf(x, p, a, b, loc=loc, scale=scale)
        assert_allclose(cdf, ref, rtol=5e-12)

    @pytest.mark.parametrize('x, p, a, b, loc, scale, ref', [(0, 1e-06, 12, -1, 0, 1, 0.38520358671350524), (-1, 3, 2.5, 2.375, 1, 3, 0.9999901774267577), (-20, 3, 2.5, 2.375, 1, 3, 1.0), (25, 2, 3, 1.5, 0.5, 1.5, 8.593419916523976e-10), (300, 10, 1.5, 0.25, 1, 5, 6.137415609872158e-24), (60, -0.125, 1000, 999, 0, 1, 2.4524915075944173e-12), (75, -0.125, 1000, 999, 0, 1, 2.9435194886214633e-18)])
    def test_sf_mpmath(self, x, p, a, b, loc, scale, ref):
        sf = stats.genhyperbolic.sf(x, p, a, b, loc=loc, scale=scale)
        assert_allclose(sf, ref, rtol=5e-12)

    def test_moments_r(self):
        vals_R = [2.36848366948115, 8.4739346779246, 37.8870502710066, 205.76608511485]
        lmbda, alpha, beta = (2, 2, 1)
        mu, delta = (0.5, 1.5)
        args = (lmbda, alpha * delta, beta * delta)
        vals_us = [stats.genhyperbolic(*args, loc=mu, scale=delta).moment(i) for i in range(1, 5)]
        assert_allclose(vals_us, vals_R, atol=0, rtol=1e-13)

    def test_rvs(self):
        lmbda, alpha, beta = (2, 2, 1)
        mu, delta = (0.5, 1.5)
        args = (lmbda, alpha * delta, beta * delta)
        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        _, p = stats.kstest(gh.rvs(size=1500, random_state=1234), gh.cdf)
        assert_equal(p > 0.05, True)

    def test_pdf_t(self):
        df = np.linspace(1, 30, 10)
        alpha, beta = (np.float_power(df, 2) * np.finfo(np.float32).eps, 0)
        mu, delta = (0, np.sqrt(df))
        args = (-df / 2, alpha, beta)
        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        x = np.linspace(gh.ppf(0.01), gh.ppf(0.99), 50)[:, np.newaxis]
        assert_allclose(gh.pdf(x), stats.t.pdf(x, df), atol=0, rtol=1e-06)

    def test_pdf_cauchy(self):
        lmbda, alpha, beta = (-0.5, np.finfo(np.float32).eps, 0)
        mu, delta = (0, 1)
        args = (lmbda, alpha, beta)
        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        x = np.linspace(gh.ppf(0.01), gh.ppf(0.99), 50)[:, np.newaxis]
        assert_allclose(gh.pdf(x), stats.cauchy.pdf(x), atol=0, rtol=1e-06)

    def test_pdf_laplace(self):
        loc = np.linspace(-10, 10, 10)
        delta = np.finfo(np.float32).eps
        lmbda, alpha, beta = (1, 1, 0)
        args = (lmbda, alpha * delta, beta * delta)
        gh = stats.genhyperbolic(*args, loc=loc, scale=delta)
        x = np.linspace(-20, 20, 50)[:, np.newaxis]
        assert_allclose(gh.pdf(x), stats.laplace.pdf(x, loc=loc, scale=1), atol=0, rtol=1e-11)

    def test_pdf_norminvgauss(self):
        alpha, beta, delta, mu = (np.linspace(1, 20, 10), np.linspace(0, 19, 10) * np.float_power(-1, range(10)), np.linspace(1, 1, 10), np.linspace(-100, 100, 10))
        lmbda = -0.5
        args = (lmbda, alpha * delta, beta * delta)
        gh = stats.genhyperbolic(*args, loc=mu, scale=delta)
        x = np.linspace(gh.ppf(0.01), gh.ppf(0.99), 50)[:, np.newaxis]
        assert_allclose(gh.pdf(x), stats.norminvgauss.pdf(x, a=alpha, b=beta, loc=mu, scale=delta), atol=0, rtol=1e-13)