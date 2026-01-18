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
class TestHypergeom:

    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.hypergeom.rvs(20, 10, 3, size=(2, 50))
        assert_(numpy.all(vals >= 0) & numpy.all(vals <= 3))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.hypergeom.rvs(20, 3, 10)
        assert_(isinstance(val, int))
        val = stats.hypergeom(20, 3, 10).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_precision(self):
        M = 2500
        n = 50
        N = 500
        tot = M
        good = n
        hgpmf = stats.hypergeom.pmf(2, tot, good, N)
        assert_almost_equal(hgpmf, 0.0010114963068932233, 11)

    def test_args(self):
        assert_almost_equal(stats.hypergeom.pmf(0, 2, 1, 0), 1.0, 11)
        assert_almost_equal(stats.hypergeom.pmf(1, 2, 1, 0), 0.0, 11)
        assert_almost_equal(stats.hypergeom.pmf(0, 2, 0, 2), 1.0, 11)
        assert_almost_equal(stats.hypergeom.pmf(1, 2, 1, 0), 0.0, 11)

    def test_cdf_above_one(self):
        assert_(0 <= stats.hypergeom.cdf(30, 13397950, 4363, 12390) <= 1.0)

    def test_precision2(self):
        oranges = 99000.0
        pears = 110000.0
        fruits_eaten = np.array([3, 3.8, 3.9, 4, 4.1, 4.2, 5]) * 10000.0
        quantile = 20000.0
        res = [stats.hypergeom.sf(quantile, oranges + pears, oranges, eaten) for eaten in fruits_eaten]
        expected = np.array([0, 1.904153e-114, 2.752693e-66, 4.931217e-32, 8.265601e-11, 0.1237904, 1])
        assert_allclose(res, expected, atol=0, rtol=5e-07)
        quantiles = [19000.0, 20000.0, 21000.0, 21500.0]
        res2 = stats.hypergeom.sf(quantiles, oranges + pears, oranges, 42000.0)
        expected2 = [1, 0.1237904, 6.511452e-34, 3.277667e-69]
        assert_allclose(res2, expected2, atol=0, rtol=5e-07)

    def test_entropy(self):
        hg = stats.hypergeom(4, 1, 1)
        h = hg.entropy()
        expected_p = np.array([0.75, 0.25])
        expected_h = -np.sum(xlogy(expected_p, expected_p))
        assert_allclose(h, expected_h)
        hg = stats.hypergeom(1, 1, 1)
        h = hg.entropy()
        assert_equal(h, 0.0)

    def test_logsf(self):
        k = 10000.0
        M = 10000000.0
        n = 1000000.0
        N = 50000.0
        result = stats.hypergeom.logsf(k, M, n, N)
        expected = -2239.771
        assert_almost_equal(result, expected, decimal=3)
        k = 1
        M = 1600
        n = 600
        N = 300
        result = stats.hypergeom.logsf(k, M, n, N)
        expected = -2.566567e-68
        assert_almost_equal(result, expected, decimal=15)

    def test_logcdf(self):
        k = 1
        M = 10000000.0
        n = 1000000.0
        N = 50000.0
        result = stats.hypergeom.logcdf(k, M, n, N)
        expected = -5273.335
        assert_almost_equal(result, expected, decimal=3)
        k = 40
        M = 1600
        n = 50
        N = 300
        result = stats.hypergeom.logcdf(k, M, n, N)
        expected = -7.565148879229e-23
        assert_almost_equal(result, expected, decimal=15)
        k = 125
        M = 1600
        n = 250
        N = 500
        result = stats.hypergeom.logcdf(k, M, n, N)
        expected = -4.242688e-12
        assert_almost_equal(result, expected, decimal=15)
        k = np.array([40, 40, 40])
        M = 1600
        n = 50
        N = 300
        result = stats.hypergeom.logcdf(k, M, n, N)
        expected = np.full(3, -7.565148879229e-23)
        assert_almost_equal(result, expected, decimal=15)

    def test_mean_gh18511(self):
        M = 390000
        n = 370000
        N = 12000
        hm = stats.hypergeom.mean(M, n, N)
        rm = n / M * N
        assert_allclose(hm, rm)

    def test_sf_gh18506(self):
        n = 10
        N = 10 ** 5
        i = np.arange(5, 15)
        population_size = 10.0 ** i
        p = stats.hypergeom.sf(n - 1, population_size, N, n)
        assert np.all(p > 0)
        assert np.all(np.diff(p) < 0)