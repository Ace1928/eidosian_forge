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
class TestRvDiscrete:

    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        states = [-1, 0, 1, 2, 3, 4]
        probability = [0.0, 0.3, 0.4, 0.0, 0.3, 0.0]
        samples = 1000
        r = stats.rv_discrete(name='sample', values=(states, probability))
        x = r.rvs(size=samples)
        assert_(isinstance(x, numpy.ndarray))
        for s, p in zip(states, probability):
            assert_(abs(sum(x == s) / float(samples) - p) < 0.05)
        x = r.rvs()
        assert np.issubdtype(type(x), np.integer)

    def test_entropy(self):
        pvals = np.array([0.25, 0.45, 0.3])
        p = stats.rv_discrete(values=([0, 1, 2], pvals))
        expected_h = -sum(xlogy(pvals, pvals))
        h = p.entropy()
        assert_allclose(h, expected_h)
        p = stats.rv_discrete(values=([0, 1, 2], [1.0, 0, 0]))
        h = p.entropy()
        assert_equal(h, 0.0)

    def test_pmf(self):
        xk = [1, 2, 4]
        pk = [0.5, 0.3, 0.2]
        rv = stats.rv_discrete(values=(xk, pk))
        x = [[1.0, 4.0], [3.0, 2]]
        assert_allclose(rv.pmf(x), [[0.5, 0.2], [0.0, 0.3]], atol=1e-14)

    def test_cdf(self):
        xk = [1, 2, 4]
        pk = [0.5, 0.3, 0.2]
        rv = stats.rv_discrete(values=(xk, pk))
        x_values = [-2, 1.0, 1.1, 1.5, 2.0, 3.0, 4, 5]
        expected = [0, 0.5, 0.5, 0.5, 0.8, 0.8, 1, 1]
        assert_allclose(rv.cdf(x_values), expected, atol=1e-14)
        assert_allclose([rv.cdf(xx) for xx in x_values], expected, atol=1e-14)

    def test_ppf(self):
        xk = [1, 2, 4]
        pk = [0.5, 0.3, 0.2]
        rv = stats.rv_discrete(values=(xk, pk))
        q_values = [0.1, 0.5, 0.6, 0.8, 0.9, 1.0]
        expected = [1, 1, 2, 2, 4, 4]
        assert_allclose(rv.ppf(q_values), expected, atol=1e-14)
        assert_allclose([rv.ppf(q) for q in q_values], expected, atol=1e-14)

    def test_cdf_ppf_next(self):
        vals = ([1, 2, 4, 7, 8], [0.1, 0.2, 0.3, 0.3, 0.1])
        rv = stats.rv_discrete(values=vals)
        assert_array_equal(rv.ppf(rv.cdf(rv.xk[:-1]) + 1e-08), rv.xk[1:])

    def test_multidimension(self):
        xk = np.arange(12).reshape((3, 4))
        pk = np.array([[0.1, 0.1, 0.15, 0.05], [0.1, 0.1, 0.05, 0.05], [0.1, 0.1, 0.05, 0.05]])
        rv = stats.rv_discrete(values=(xk, pk))
        assert_allclose(rv.expect(), np.sum(rv.xk * rv.pk), atol=1e-14)

    def test_bad_input(self):
        xk = [1, 2, 3]
        pk = [0.5, 0.5]
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))
        pk = [1, 2, 3]
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))
        xk = [1, 2, 3]
        pk = [0.5, 1.2, -0.7]
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))
        xk = [1, 2, 3, 4, 5]
        pk = [0.3, 0.3, 0.3, 0.3, -0.2]
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))
        xk = [1, 1]
        pk = [0.5, 0.5]
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))

    def test_shape_rv_sample(self):
        xk, pk = (np.arange(4).reshape((2, 2)), np.full((2, 3), 1 / 6))
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))
        xk, pk = (np.arange(6).reshape((3, 2)), np.full((2, 3), 1 / 6))
        assert_raises(ValueError, stats.rv_discrete, **dict(values=(xk, pk)))
        xk, pk = (np.arange(6).reshape((3, 2)), np.full((3, 2), 1 / 6))
        assert_equal(stats.rv_discrete(values=(xk, pk)).pmf(0), 1 / 6)

    def test_expect1(self):
        xk = [1, 2, 4, 6, 7, 11]
        pk = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
        rv = stats.rv_discrete(values=(xk, pk))
        assert_allclose(rv.expect(), np.sum(rv.xk * rv.pk), atol=1e-14)

    def test_expect2(self):
        y = [200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0, 3000.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0, 4000.0, 4100.0, 4200.0, 4300.0, 4400.0, 4500.0, 4600.0, 4700.0, 4800.0]
        py = [0.0004, 0.0, 0.0033, 0.006500000000000001, 0.0, 0.0, 0.004399999999999999, 0.6862, 0.0, 0.0, 0.0, 0.00019999999999997797, 0.0006000000000000449, 0.024499999999999966, 0.006400000000000072, 0.0043999999999999595, 0.019499999999999962, 0.03770000000000007, 0.01759999999999995, 0.015199999999999991, 0.018100000000000005, 0.04500000000000004, 0.0025999999999999357, 0.0, 0.0041000000000001036, 0.005999999999999894, 0.0042000000000000925, 0.0050000000000000044, 0.0041999999999999815, 0.0004999999999999449, 0.009199999999999986, 0.008200000000000096, 0.0, 0.0, 0.0046999999999999265, 0.0019000000000000128, 0.0006000000000000449, 0.02510000000000001, 0.0, 0.007199999999999984, 0.0, 0.012699999999999934, 0.0, 0.0, 0.008199999999999985, 0.005600000000000049, 0.0]
        rv = stats.rv_discrete(values=(y, py))
        assert_allclose(rv.expect(), rv.mean(), atol=1e-14)
        assert_allclose(rv.expect(), sum((v * w for v, w in zip(y, py))), atol=1e-14)
        assert_allclose(rv.expect(lambda x: x ** 2), sum((v ** 2 * w for v, w in zip(y, py))), atol=1e-14)