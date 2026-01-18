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
class TestGenInvGauss:

    def setup_method(self):
        np.random.seed(1234)

    @pytest.mark.slow
    def test_rvs_with_mode_shift(self):
        gig = stats.geninvgauss(2.3, 1.5)
        _, p = stats.kstest(gig.rvs(size=1500, random_state=1234), gig.cdf)
        assert_equal(p > 0.05, True)

    @pytest.mark.slow
    def test_rvs_without_mode_shift(self):
        gig = stats.geninvgauss(0.9, 0.75)
        _, p = stats.kstest(gig.rvs(size=1500, random_state=1234), gig.cdf)
        assert_equal(p > 0.05, True)

    @pytest.mark.slow
    def test_rvs_new_method(self):
        gig = stats.geninvgauss(0.1, 0.2)
        _, p = stats.kstest(gig.rvs(size=1500, random_state=1234), gig.cdf)
        assert_equal(p > 0.05, True)

    @pytest.mark.slow
    def test_rvs_p_zero(self):

        def my_ks_check(p, b):
            gig = stats.geninvgauss(p, b)
            rvs = gig.rvs(size=1500, random_state=1234)
            return stats.kstest(rvs, gig.cdf)[1] > 0.05
        assert_equal(my_ks_check(0, 0.2), True)
        assert_equal(my_ks_check(0, 0.9), True)
        assert_equal(my_ks_check(0, 1.5), True)

    def test_rvs_negative_p(self):
        assert_equal(stats.geninvgauss(-1.5, 2).rvs(size=10, random_state=1234), 1 / stats.geninvgauss(1.5, 2).rvs(size=10, random_state=1234))

    def test_invgauss(self):
        ig = stats.geninvgauss.rvs(size=1500, p=-0.5, b=1, random_state=1234)
        assert_equal(stats.kstest(ig, 'invgauss', args=[1])[1] > 0.15, True)
        mu, x = (100, np.linspace(0.01, 1, 10))
        pdf_ig = stats.geninvgauss.pdf(x, p=-0.5, b=1 / mu, scale=mu)
        assert_allclose(pdf_ig, stats.invgauss(mu).pdf(x))
        cdf_ig = stats.geninvgauss.cdf(x, p=-0.5, b=1 / mu, scale=mu)
        assert_allclose(cdf_ig, stats.invgauss(mu).cdf(x))

    def test_pdf_R(self):
        vals_R = np.array([2.08117682e-21, 0.4488660034, 0.3747774338, 0.2693297528, 0.1905637275, 0.1351476913, 0.09636538981, 0.06909040154, 0.04978006801, 0.03602084467])
        x = np.linspace(0.01, 5, 10)
        assert_allclose(vals_R, stats.geninvgauss.pdf(x, 0.5, 1))

    def test_pdf_zero(self):
        assert_equal(stats.geninvgauss.pdf(0, 0.5, 0.5), 0)
        assert_equal(stats.geninvgauss.pdf(2000000.0, 50, 2), 0)