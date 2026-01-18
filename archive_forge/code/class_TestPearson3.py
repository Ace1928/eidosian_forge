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
class TestPearson3:

    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.pearson3.rvs(0.1, size=(2, 50))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllFloat'])
        val = stats.pearson3.rvs(0.5)
        assert_(isinstance(val, float))
        val = stats.pearson3(0.5).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllFloat'])
        assert_(len(val) == 3)

    def test_pdf(self):
        vals = stats.pearson3.pdf(2, [0.0, 0.1, 0.2])
        assert_allclose(vals, np.array([0.05399097, 0.05555481, 0.05670246]), atol=1e-06)
        vals = stats.pearson3.pdf(-3, 0.1)
        assert_allclose(vals, np.array([0.00313791]), atol=1e-06)
        vals = stats.pearson3.pdf([-3, -2, -1, 0, 1], 0.1)
        assert_allclose(vals, np.array([0.00313791, 0.05192304, 0.25028092, 0.39885918, 0.23413173]), atol=1e-06)

    def test_cdf(self):
        vals = stats.pearson3.cdf(2, [0.0, 0.1, 0.2])
        assert_allclose(vals, np.array([0.97724987, 0.97462004, 0.97213626]), atol=1e-06)
        vals = stats.pearson3.cdf(-3, 0.1)
        assert_allclose(vals, [0.00082256], atol=1e-06)
        vals = stats.pearson3.cdf([-3, -2, -1, 0, 1], 0.1)
        assert_allclose(vals, [0.000822563821, 0.0199860448, 0.15855071, 0.50664913, 0.841442111], atol=1e-06)

    def test_negative_cdf_bug_11186(self):
        skews = [-3, -1, 0, 0.5]
        x_eval = 0.5
        neg_inf = -30
        cdfs = stats.pearson3.cdf(x_eval, skews)
        int_pdfs = [quad(stats.pearson3(skew).pdf, neg_inf, x_eval)[0] for skew in skews]
        assert_allclose(cdfs, int_pdfs)

    def test_return_array_bug_11746(self):
        moment = stats.pearson3.moment(1, 2)
        assert_equal(moment, 0)
        assert isinstance(moment, np.number)
        moment = stats.pearson3.moment(1, 1e-06)
        assert_equal(moment, 0)
        assert isinstance(moment, np.number)

    def test_ppf_bug_17050(self):
        skews = [-3, -1, 0, 0.5]
        x_eval = 0.5
        res = stats.pearson3.ppf(stats.pearson3.cdf(x_eval, skews), skews)
        assert_allclose(res, x_eval)
        skew = np.array([[-0.5], [1.5]])
        x = np.linspace(-2, 2)
        assert_allclose(stats.pearson3.pdf(x, skew), stats.pearson3.pdf(-x, -skew))
        assert_allclose(stats.pearson3.cdf(x, skew), stats.pearson3.sf(-x, -skew))
        assert_allclose(stats.pearson3.ppf(x, skew), -stats.pearson3.isf(x, -skew))

    def test_sf(self):
        skew = [0.1, 0.5, 1.0, -0.1]
        x = [5.0, 10.0, 50.0, 8.0]
        ref = [1.64721926440872e-06, 8.271911573556123e-11, 1.3149506021756343e-40, 2.763057937820296e-21]
        assert_allclose(stats.pearson3.sf(x, skew), ref, rtol=2e-14)
        assert_allclose(stats.pearson3.sf(x, 0), stats.norm.sf(x), rtol=2e-14)