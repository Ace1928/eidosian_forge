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
class TestFitMethod:
    skip = ['ncf', 'ksone', 'kstwo']

    def setup_method(self):
        np.random.seed(1234)
    fitSkipNonFinite = ['expon', 'norm', 'uniform']

    @pytest.mark.parametrize('dist,args', distcont)
    def test_fit_w_non_finite_data_values(self, dist, args):
        """gh-10300"""
        if dist in self.fitSkipNonFinite:
            pytest.skip('%s fit known to fail or deprecated' % dist)
        x = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.nan])
        y = np.array([1.6483, 2.7169, 2.4667, 1.1791, 3.5433, np.inf])
        distfunc = getattr(stats, dist)
        assert_raises(ValueError, distfunc.fit, x, fscale=1)
        assert_raises(ValueError, distfunc.fit, y, fscale=1)

    def test_fix_fit_2args_lognorm(self):
        np.random.seed(12345)
        with np.errstate(all='ignore'):
            x = stats.lognorm.rvs(0.25, 0.0, 20.0, size=20)
            expected_shape = np.sqrt(((np.log(x) - np.log(20)) ** 2).mean())
            assert_allclose(np.array(stats.lognorm.fit(x, floc=0, fscale=20)), [expected_shape, 0, 20], atol=1e-08)

    def test_fix_fit_norm(self):
        x = np.arange(1, 6)
        loc, scale = stats.norm.fit(x)
        assert_almost_equal(loc, 3)
        assert_almost_equal(scale, np.sqrt(2))
        loc, scale = stats.norm.fit(x, floc=2)
        assert_equal(loc, 2)
        assert_equal(scale, np.sqrt(3))
        loc, scale = stats.norm.fit(x, fscale=2)
        assert_almost_equal(loc, 3)
        assert_equal(scale, 2)

    def test_fix_fit_gamma(self):
        x = np.arange(1, 6)
        meanlog = np.log(x).mean()
        floc = 0
        a, loc, scale = stats.gamma.fit(x, floc=floc)
        s = np.log(x.mean()) - meanlog
        assert_almost_equal(np.log(a) - special.digamma(a), s, decimal=5)
        assert_equal(loc, floc)
        assert_almost_equal(scale, x.mean() / a, decimal=8)
        f0 = 1
        floc = 0
        a, loc, scale = stats.gamma.fit(x, f0=f0, floc=floc)
        assert_equal(a, f0)
        assert_equal(loc, floc)
        assert_almost_equal(scale, x.mean() / a, decimal=8)
        f0 = 2
        floc = 0
        a, loc, scale = stats.gamma.fit(x, f0=f0, floc=floc)
        assert_equal(a, f0)
        assert_equal(loc, floc)
        assert_almost_equal(scale, x.mean() / a, decimal=8)
        floc = 0
        fscale = 2
        a, loc, scale = stats.gamma.fit(x, floc=floc, fscale=fscale)
        assert_equal(loc, floc)
        assert_equal(scale, fscale)
        c = meanlog - np.log(fscale)
        assert_almost_equal(special.digamma(a), c)

    def test_fix_fit_beta(self):

        def mlefunc(a, b, x):
            n = len(x)
            s1 = np.log(x).sum()
            s2 = np.log(1 - x).sum()
            psiab = special.psi(a + b)
            func = [s1 - n * (-psiab + special.psi(a)), s2 - n * (-psiab + special.psi(b))]
            return func
        x = np.array([0.125, 0.25, 0.5])
        a, b, loc, scale = stats.beta.fit(x, floc=0, fscale=1)
        assert_equal(loc, 0)
        assert_equal(scale, 1)
        assert_allclose(mlefunc(a, b, x), [0, 0], atol=1e-06)
        x = np.array([0.125, 0.25, 0.5])
        a, b, loc, scale = stats.beta.fit(x, f0=2, floc=0, fscale=1)
        assert_equal(a, 2)
        assert_equal(loc, 0)
        assert_equal(scale, 1)
        da, db = mlefunc(a, b, x)
        assert_allclose(db, 0, atol=1e-05)
        x2 = 1 - x
        a2, b2, loc2, scale2 = stats.beta.fit(x2, f1=2, floc=0, fscale=1)
        assert_equal(b2, 2)
        assert_equal(loc2, 0)
        assert_equal(scale2, 1)
        da, db = mlefunc(a2, b2, x2)
        assert_allclose(da, 0, atol=1e-05)
        assert_almost_equal(a2, b)
        assert_raises(ValueError, stats.beta.fit, x, floc=0.5, fscale=1)
        y = np.array([0, 0.5, 1])
        assert_raises(ValueError, stats.beta.fit, y, floc=0, fscale=1)
        assert_raises(ValueError, stats.beta.fit, y, floc=0, fscale=1, f0=2)
        assert_raises(ValueError, stats.beta.fit, y, floc=0, fscale=1, f1=2)
        assert_raises(ValueError, stats.beta.fit, y, f0=0, f1=1, floc=2, fscale=3)

    def test_expon_fit(self):
        x = np.array([2, 2, 4, 4, 4, 4, 4, 8])
        loc, scale = stats.expon.fit(x)
        assert_equal(loc, 2)
        assert_equal(scale, 2)
        loc, scale = stats.expon.fit(x, fscale=3)
        assert_equal(loc, 2)
        assert_equal(scale, 3)
        loc, scale = stats.expon.fit(x, floc=0)
        assert_equal(loc, 0)
        assert_equal(scale, 4)

    def test_lognorm_fit(self):
        x = np.array([1.5, 3, 10, 15, 23, 59])
        lnxm1 = np.log(x - 1)
        shape, loc, scale = stats.lognorm.fit(x, floc=1)
        assert_allclose(shape, lnxm1.std(), rtol=1e-12)
        assert_equal(loc, 1)
        assert_allclose(scale, np.exp(lnxm1.mean()), rtol=1e-12)
        shape, loc, scale = stats.lognorm.fit(x, floc=1, fscale=6)
        assert_allclose(shape, np.sqrt(((lnxm1 - np.log(6)) ** 2).mean()), rtol=1e-12)
        assert_equal(loc, 1)
        assert_equal(scale, 6)
        shape, loc, scale = stats.lognorm.fit(x, floc=1, fix_s=0.75)
        assert_equal(shape, 0.75)
        assert_equal(loc, 1)
        assert_allclose(scale, np.exp(lnxm1.mean()), rtol=1e-12)

    def test_uniform_fit(self):
        x = np.array([1.0, 1.1, 1.2, 9.0])
        loc, scale = stats.uniform.fit(x)
        assert_equal(loc, x.min())
        assert_equal(scale, np.ptp(x))
        loc, scale = stats.uniform.fit(x, floc=0)
        assert_equal(loc, 0)
        assert_equal(scale, x.max())
        loc, scale = stats.uniform.fit(x, fscale=10)
        assert_equal(loc, 0)
        assert_equal(scale, 10)
        assert_raises(ValueError, stats.uniform.fit, x, floc=2.0)
        assert_raises(ValueError, stats.uniform.fit, x, fscale=5.0)

    @pytest.mark.slow
    @pytest.mark.parametrize('method', ['MLE', 'MM'])
    def test_fshapes(self, method):
        a, b = (3.0, 4.0)
        x = stats.beta.rvs(a, b, size=100, random_state=1234)
        res_1 = stats.beta.fit(x, f0=3.0, method=method)
        res_2 = stats.beta.fit(x, fa=3.0, method=method)
        assert_allclose(res_1, res_2, atol=1e-12, rtol=1e-12)
        res_2 = stats.beta.fit(x, fix_a=3.0, method=method)
        assert_allclose(res_1, res_2, atol=1e-12, rtol=1e-12)
        res_3 = stats.beta.fit(x, f1=4.0, method=method)
        res_4 = stats.beta.fit(x, fb=4.0, method=method)
        assert_allclose(res_3, res_4, atol=1e-12, rtol=1e-12)
        res_4 = stats.beta.fit(x, fix_b=4.0, method=method)
        assert_allclose(res_3, res_4, atol=1e-12, rtol=1e-12)
        assert_raises(ValueError, stats.beta.fit, x, fa=1, f0=2, method=method)
        assert_raises(ValueError, stats.beta.fit, x, fa=0, f1=1, floc=2, fscale=3, method=method)
        res_5 = stats.beta.fit(x, fa=3.0, floc=0, fscale=1, method=method)
        aa, bb, ll, ss = res_5
        assert_equal([aa, ll, ss], [3.0, 0, 1])
        a = 3.0
        data = stats.gamma.rvs(a, size=100)
        aa, ll, ss = stats.gamma.fit(data, fa=a, method=method)
        assert_equal(aa, a)

    @pytest.mark.parametrize('method', ['MLE', 'MM'])
    def test_extra_params(self, method):
        dist = stats.exponnorm
        data = dist.rvs(K=2, size=100)
        dct = dict(enikibeniki=-101)
        assert_raises(TypeError, dist.fit, data, **dct, method=method)