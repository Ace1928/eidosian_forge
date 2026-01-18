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
class TestFrozen:

    def setup_method(self):
        np.random.seed(1234)

    def test_norm(self):
        dist = stats.norm
        frozen = stats.norm(loc=10.0, scale=3.0)
        result_f = frozen.pdf(20.0)
        result = dist.pdf(20.0, loc=10.0, scale=3.0)
        assert_equal(result_f, result)
        result_f = frozen.cdf(20.0)
        result = dist.cdf(20.0, loc=10.0, scale=3.0)
        assert_equal(result_f, result)
        result_f = frozen.ppf(0.25)
        result = dist.ppf(0.25, loc=10.0, scale=3.0)
        assert_equal(result_f, result)
        result_f = frozen.isf(0.25)
        result = dist.isf(0.25, loc=10.0, scale=3.0)
        assert_equal(result_f, result)
        result_f = frozen.sf(10.0)
        result = dist.sf(10.0, loc=10.0, scale=3.0)
        assert_equal(result_f, result)
        result_f = frozen.median()
        result = dist.median(loc=10.0, scale=3.0)
        assert_equal(result_f, result)
        result_f = frozen.mean()
        result = dist.mean(loc=10.0, scale=3.0)
        assert_equal(result_f, result)
        result_f = frozen.var()
        result = dist.var(loc=10.0, scale=3.0)
        assert_equal(result_f, result)
        result_f = frozen.std()
        result = dist.std(loc=10.0, scale=3.0)
        assert_equal(result_f, result)
        result_f = frozen.entropy()
        result = dist.entropy(loc=10.0, scale=3.0)
        assert_equal(result_f, result)
        result_f = frozen.moment(2)
        result = dist.moment(2, loc=10.0, scale=3.0)
        assert_equal(result_f, result)
        assert_equal(frozen.a, dist.a)
        assert_equal(frozen.b, dist.b)

    def test_gamma(self):
        a = 2.0
        dist = stats.gamma
        frozen = stats.gamma(a)
        result_f = frozen.pdf(20.0)
        result = dist.pdf(20.0, a)
        assert_equal(result_f, result)
        result_f = frozen.cdf(20.0)
        result = dist.cdf(20.0, a)
        assert_equal(result_f, result)
        result_f = frozen.ppf(0.25)
        result = dist.ppf(0.25, a)
        assert_equal(result_f, result)
        result_f = frozen.isf(0.25)
        result = dist.isf(0.25, a)
        assert_equal(result_f, result)
        result_f = frozen.sf(10.0)
        result = dist.sf(10.0, a)
        assert_equal(result_f, result)
        result_f = frozen.median()
        result = dist.median(a)
        assert_equal(result_f, result)
        result_f = frozen.mean()
        result = dist.mean(a)
        assert_equal(result_f, result)
        result_f = frozen.var()
        result = dist.var(a)
        assert_equal(result_f, result)
        result_f = frozen.std()
        result = dist.std(a)
        assert_equal(result_f, result)
        result_f = frozen.entropy()
        result = dist.entropy(a)
        assert_equal(result_f, result)
        result_f = frozen.moment(2)
        result = dist.moment(2, a)
        assert_equal(result_f, result)
        assert_equal(frozen.a, frozen.dist.a)
        assert_equal(frozen.b, frozen.dist.b)

    def test_regression_ticket_1293(self):
        frozen = stats.lognorm(1)
        m1 = frozen.moment(2)
        frozen.stats(moments='mvsk')
        m2 = frozen.moment(2)
        assert_equal(m1, m2)

    def test_ab(self):
        c = -0.1
        rv = stats.genpareto(c=c)
        a, b = rv.dist._get_support(c)
        assert_equal([a, b], [0.0, 10.0])
        c = 0.1
        stats.genpareto.pdf(0, c=c)
        assert_equal(rv.dist._get_support(c), [0, np.inf])
        c = -0.1
        rv = stats.genpareto(c=c)
        a, b = rv.dist._get_support(c)
        assert_equal([a, b], [0.0, 10.0])
        c = 0.1
        stats.genpareto.pdf(0, c)
        assert_equal((rv.dist.a, rv.dist.b), stats.genpareto._get_support(c))
        rv1 = stats.genpareto(c=0.1)
        assert_(rv1.dist is not rv.dist)
        for c in [1.0, 0.0]:
            c = np.asarray(c)
            rv = stats.genpareto(c=c)
            a, b = (rv.a, rv.b)
            assert_equal(a, 0.0)
            assert_(np.isposinf(b))
            c = np.asarray(-2.0)
            a, b = stats.genpareto._get_support(c)
            assert_allclose([a, b], [0.0, 0.5])

    def test_rv_frozen_in_namespace(self):
        assert_(hasattr(stats.distributions, 'rv_frozen'))

    def test_random_state(self):
        frozen = stats.norm()
        assert_(hasattr(frozen, 'random_state'))
        frozen.random_state = 42
        assert_equal(frozen.random_state.get_state(), np.random.RandomState(42).get_state())
        rndm = np.random.RandomState(1234)
        frozen.rvs(size=8, random_state=rndm)

    def test_pickling(self):
        beta = stats.beta(2.3098496451481823, 0.6268795430096368)
        poiss = stats.poisson(3.0)
        sample = stats.rv_discrete(values=([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]))
        for distfn in [beta, poiss, sample]:
            distfn.random_state = 1234
            distfn.rvs(size=8)
            s = pickle.dumps(distfn)
            r0 = distfn.rvs(size=8)
            unpickled = pickle.loads(s)
            r1 = unpickled.rvs(size=8)
            assert_equal(r0, r1)
            medians = [distfn.ppf(0.5), unpickled.ppf(0.5)]
            assert_equal(medians[0], medians[1])
            assert_equal(distfn.cdf(medians[0]), unpickled.cdf(medians[1]))

    def test_expect(self):

        def func(x):
            return x
        gm = stats.gamma(a=2, loc=3, scale=4)
        with np.errstate(invalid='ignore', divide='ignore'):
            gm_val = gm.expect(func, lb=1, ub=2, conditional=True)
            gamma_val = stats.gamma.expect(func, args=(2,), loc=3, scale=4, lb=1, ub=2, conditional=True)
        assert_allclose(gm_val, gamma_val)
        p = stats.poisson(3, loc=4)
        p_val = p.expect(func)
        poisson_val = stats.poisson.expect(func, args=(3,), loc=4)
        assert_allclose(p_val, poisson_val)