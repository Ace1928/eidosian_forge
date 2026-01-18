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
class TestJFSkewT:

    def test_compare_t(self):
        a = b = 5
        df = a * 2
        x = [-1.0, 0.0, 1.0, 2.0]
        q = [0.0, 0.1, 0.25, 0.75, 0.9, 1.0]
        jf = stats.jf_skew_t(a, b)
        t = stats.t(df)
        assert_allclose(jf.pdf(x), t.pdf(x))
        assert_allclose(jf.cdf(x), t.cdf(x))
        assert_allclose(jf.ppf(q), t.ppf(q))
        assert_allclose(jf.stats('mvsk'), t.stats('mvsk'))

    @pytest.fixture
    def gamlss_pdf_data(self):
        """Sample data points computed using the `ST5` distribution from the
        GAMLSS package in R. The pdf has been calculated for (a,b)=(2,3),
        (a,b)=(8,4), and (a,b)=(12,13) for x in `np.linspace(-10, 10, 41)`.

        N.B. the `ST5` distribution in R uses an alternative parameterization
        in terms of nu and tau, where:
            - nu = (a - b) / (a * b * (a + b)) ** 0.5
            - tau = 2 / (a + b)
        """
        data = np.load(Path(__file__).parent / 'data/jf_skew_t_gamlss_pdf_data.npy')
        return np.rec.fromarrays(data, names='x,pdf,a,b')

    @pytest.mark.parametrize('a,b', [(2, 3), (8, 4), (12, 13)])
    def test_compare_with_gamlss_r(self, gamlss_pdf_data, a, b):
        """Compare the pdf with a table of reference values. The table of
        reference values was produced using R, where the Jones and Faddy skew
        t distribution is available in the GAMLSS package as `ST5`.
        """
        data = gamlss_pdf_data[(gamlss_pdf_data['a'] == a) & (gamlss_pdf_data['b'] == b)]
        x, pdf = (data['x'], data['pdf'])
        assert_allclose(pdf, stats.jf_skew_t(a, b).pdf(x), rtol=1e-12)