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
class TestInvgauss:

    def setup_method(self):
        np.random.seed(1234)

    @pytest.mark.parametrize('rvs_mu,rvs_loc,rvs_scale', [(2, 0, 1), (4.635, 4.362, 6.303)])
    def test_fit(self, rvs_mu, rvs_loc, rvs_scale):
        data = stats.invgauss.rvs(size=100, mu=rvs_mu, loc=rvs_loc, scale=rvs_scale)
        mu, loc, scale = stats.invgauss.fit(data, floc=rvs_loc)
        data = data - rvs_loc
        mu_temp = np.mean(data)
        scale_mle = len(data) / np.sum(data ** (-1) - mu_temp ** (-1))
        mu_mle = mu_temp / scale_mle
        assert_allclose(mu_mle, mu, atol=1e-15, rtol=1e-15)
        assert_allclose(scale_mle, scale, atol=1e-15, rtol=1e-15)
        assert_equal(loc, rvs_loc)
        data = stats.invgauss.rvs(size=100, mu=rvs_mu, loc=rvs_loc, scale=rvs_scale)
        mu, loc, scale = stats.invgauss.fit(data, floc=rvs_loc - 1, fscale=rvs_scale + 1)
        assert_equal(rvs_scale + 1, scale)
        assert_equal(rvs_loc - 1, loc)
        shape_mle1 = stats.invgauss.fit(data, fmu=1.04)[0]
        shape_mle2 = stats.invgauss.fit(data, fix_mu=1.04)[0]
        shape_mle3 = stats.invgauss.fit(data, f0=1.04)[0]
        assert shape_mle1 == shape_mle2 == shape_mle3 == 1.04

    @pytest.mark.parametrize('rvs_mu,rvs_loc,rvs_scale', [(2, 0, 1), (6.311, 3.225, 4.52)])
    def test_fit_MLE_comp_optimizer(self, rvs_mu, rvs_loc, rvs_scale):
        data = stats.invgauss.rvs(size=100, mu=rvs_mu, loc=rvs_loc, scale=rvs_scale)
        super_fit = super(type(stats.invgauss), stats.invgauss).fit
        super_fitted = super_fit(data)
        invgauss_fit = stats.invgauss.fit(data)
        assert_equal(super_fitted, invgauss_fit)
        super_fitted = super_fit(data, floc=0, fmu=2)
        invgauss_fit = stats.invgauss.fit(data, floc=0, fmu=2)
        assert_equal(super_fitted, invgauss_fit)
        _assert_less_or_close_loglike(stats.invgauss, data, floc=rvs_loc)
        assert np.all(data - (rvs_loc - 1) > 0)
        _assert_less_or_close_loglike(stats.invgauss, data, floc=rvs_loc - 1)
        _assert_less_or_close_loglike(stats.invgauss, data, floc=0)
        _assert_less_or_close_loglike(stats.invgauss, data, floc=rvs_loc, fscale=np.random.rand(1)[0])

    def test_fit_raise_errors(self):
        assert_fit_warnings(stats.invgauss)
        with pytest.raises(FitDataError):
            stats.invgauss.fit([1, 2, 3], floc=2)

    def test_cdf_sf(self):
        mu = [0.000417022005, 0.00720324493, 1.14374817e-06, 0.00302332573, 0.00146755891]
        expected = [1, 1, 1, 1, 1]
        actual = stats.invgauss.cdf(0.4, mu=mu)
        assert_equal(expected, actual)
        cdf_actual = stats.invgauss.cdf(0.001, mu=1.05)
        assert_allclose(cdf_actual, 4.65246506892667e-219)
        sf_actual = stats.invgauss.sf(110, mu=1.05)
        assert_allclose(sf_actual, 4.12851625944048e-25)
        actual = stats.invgauss.cdf(9e-05, 0.0001)
        assert_allclose(actual, 2.9458022894924e-26)
        actual = stats.invgauss.cdf(0.000102, 0.0001)
        assert_allclose(actual, 0.976445540507925)

    def test_logcdf_logsf(self):
        logcdf = stats.invgauss.logcdf(0.0001, mu=1.05)
        assert_allclose(logcdf, -5003.87872590367)
        logcdf = stats.invgauss.logcdf(110, 1.05)
        assert_allclose(logcdf, -4.12851625944087e-25)
        logsf = stats.invgauss.logsf(0.001, mu=1.05)
        assert_allclose(logsf, -4.65246506892676e-219)
        logsf = stats.invgauss.logsf(110, 1.05)
        assert_allclose(logsf, -56.1467092416426)

    @pytest.mark.parametrize('mu, ref', [(2e-08, -25.172361826883957), (0.001, -8.943444010642972), (0.01, -5.4962796152622335), (100000000.0, 3.3244822568873476), (1e+100, 3.32448280139689)])
    def test_entropy(self, mu, ref):
        assert_allclose(stats.invgauss.entropy(mu), ref, rtol=5e-14)