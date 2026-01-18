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
class TestHistogram:

    def setup_method(self):
        np.random.seed(1234)
        histogram = np.histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9], bins=8)
        self.template = stats.rv_histogram(histogram)
        data = stats.norm.rvs(loc=1.0, scale=2.5, size=10000, random_state=123)
        norm_histogram = np.histogram(data, bins=50)
        self.norm_template = stats.rv_histogram(norm_histogram)

    def test_pdf(self):
        values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
        pdf_values = np.asarray([0.0 / 25.0, 0.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0, 2.0 / 25.0, 2.0 / 25.0, 3.0 / 25.0, 3.0 / 25.0, 4.0 / 25.0, 4.0 / 25.0, 5.0 / 25.0, 5.0 / 25.0, 4.0 / 25.0, 4.0 / 25.0, 3.0 / 25.0, 3.0 / 25.0, 3.0 / 25.0, 3.0 / 25.0, 0.0 / 25.0, 0.0 / 25.0])
        assert_allclose(self.template.pdf(values), pdf_values)
        assert_almost_equal(self.template.pdf(8.0), 3.0 / 25.0)
        assert_almost_equal(self.template.pdf(8.5), 3.0 / 25.0)
        assert_almost_equal(self.template.pdf(9.0), 0.0 / 25.0)
        assert_almost_equal(self.template.pdf(10.0), 0.0 / 25.0)
        x = np.linspace(-2, 2, 10)
        assert_allclose(self.norm_template.pdf(x), stats.norm.pdf(x, loc=1.0, scale=2.5), rtol=0.1)

    def test_cdf_ppf(self):
        values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
        cdf_values = np.asarray([0.0 / 25.0, 0.0 / 25.0, 0.0 / 25.0, 0.5 / 25.0, 1.0 / 25.0, 2.0 / 25.0, 3.0 / 25.0, 4.5 / 25.0, 6.0 / 25.0, 8.0 / 25.0, 10.0 / 25.0, 12.5 / 25.0, 15.0 / 25.0, 17.0 / 25.0, 19.0 / 25.0, 20.5 / 25.0, 22.0 / 25.0, 23.5 / 25.0, 25.0 / 25.0, 25.0 / 25.0])
        assert_allclose(self.template.cdf(values), cdf_values)
        assert_allclose(self.template.ppf(cdf_values[2:-1]), values[2:-1])
        x = np.linspace(1.0, 9.0, 100)
        assert_allclose(self.template.ppf(self.template.cdf(x)), x)
        x = np.linspace(0.0, 1.0, 100)
        assert_allclose(self.template.cdf(self.template.ppf(x)), x)
        x = np.linspace(-2, 2, 10)
        assert_allclose(self.norm_template.cdf(x), stats.norm.cdf(x, loc=1.0, scale=2.5), rtol=0.1)

    def test_rvs(self):
        N = 10000
        sample = self.template.rvs(size=N, random_state=123)
        assert_equal(np.sum(sample < 1.0), 0.0)
        assert_allclose(np.sum(sample <= 2.0), 1.0 / 25.0 * N, rtol=0.2)
        assert_allclose(np.sum(sample <= 2.5), 2.0 / 25.0 * N, rtol=0.2)
        assert_allclose(np.sum(sample <= 3.0), 3.0 / 25.0 * N, rtol=0.1)
        assert_allclose(np.sum(sample <= 3.5), 4.5 / 25.0 * N, rtol=0.1)
        assert_allclose(np.sum(sample <= 4.0), 6.0 / 25.0 * N, rtol=0.1)
        assert_allclose(np.sum(sample <= 4.5), 8.0 / 25.0 * N, rtol=0.1)
        assert_allclose(np.sum(sample <= 5.0), 10.0 / 25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 5.5), 12.5 / 25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 6.0), 15.0 / 25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 6.5), 17.0 / 25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 7.0), 19.0 / 25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 7.5), 20.5 / 25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 8.0), 22.0 / 25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 8.5), 23.5 / 25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 9.0), 25.0 / 25.0 * N, rtol=0.05)
        assert_allclose(np.sum(sample <= 9.0), 25.0 / 25.0 * N, rtol=0.05)
        assert_equal(np.sum(sample > 9.0), 0.0)

    def test_munp(self):
        for n in range(4):
            assert_allclose(self.norm_template._munp(n), stats.norm(1.0, 2.5).moment(n), rtol=0.05)

    def test_entropy(self):
        assert_allclose(self.norm_template.entropy(), stats.norm.entropy(loc=1.0, scale=2.5), rtol=0.05)