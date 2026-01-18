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
class TestBinom:

    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.binom.rvs(10, 0.75, size=(2, 50))
        assert_(numpy.all(vals >= 0) & numpy.all(vals <= 10))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.binom.rvs(10, 0.75)
        assert_(isinstance(val, int))
        val = stats.binom(10, 0.75).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_pmf(self):
        vals1 = stats.binom.pmf(100, 100, 1)
        vals2 = stats.binom.pmf(0, 100, 0)
        assert_allclose(vals1, 1.0, rtol=1e-15, atol=0)
        assert_allclose(vals2, 1.0, rtol=1e-15, atol=0)

    def test_entropy(self):
        b = stats.binom(2, 0.5)
        expected_p = np.array([0.25, 0.5, 0.25])
        expected_h = -sum(xlogy(expected_p, expected_p))
        h = b.entropy()
        assert_allclose(h, expected_h)
        b = stats.binom(2, 0.0)
        h = b.entropy()
        assert_equal(h, 0.0)
        b = stats.binom(2, 1.0)
        h = b.entropy()
        assert_equal(h, 0.0)

    def test_warns_p0(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            assert_equal(stats.binom(n=2, p=0).mean(), 0)
            assert_equal(stats.binom(n=2, p=0).std(), 0)

    def test_ppf_p1(self):
        n = 4
        assert stats.binom.ppf(q=0.3, n=n, p=1.0) == n

    def test_pmf_poisson(self):
        n = 1541096362225563.0
        p = 1.0477878413173978e-18
        x = np.arange(3)
        res = stats.binom.pmf(x, n=n, p=p)
        ref = stats.poisson.pmf(x, n * p)
        assert_allclose(res, ref, atol=1e-16)

    def test_pmf_cdf(self):
        n = 25.0 * 10 ** 21
        p = 1.0 * 10 ** (-21)
        r = 0
        res = stats.binom.pmf(r, n, p)
        ref = stats.binom.cdf(r, n, p)
        assert_allclose(res, ref, atol=1e-16)

    def test_pmf_gh15101(self):
        res = stats.binom.pmf(3, 2000, 0.999)
        assert_allclose(res, 0, atol=1e-16)