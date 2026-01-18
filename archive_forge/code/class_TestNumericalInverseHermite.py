import threading
import pickle
import pytest
from copy import deepcopy
import platform
import sys
import math
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy.stats.sampling import (
from pytest import raises as assert_raises
from scipy import stats
from scipy import special
from scipy.stats import chisquare, cramervonmises
from scipy.stats._distr_params import distdiscrete, distcont
from scipy._lib._util import check_random_state
class TestNumericalInverseHermite:

    class dist0:

        def pdf(self, x):
            return 0.5 * (1.0 + np.sin(2.0 * np.pi * x))

        def dpdf(self, x):
            return np.pi * np.cos(2.0 * np.pi * x)

        def cdf(self, x):
            return (1.0 + 2.0 * np.pi * (1 + x) - np.cos(2.0 * np.pi * x)) / (4.0 * np.pi)

        def support(self):
            return (-1, 1)

    class dist1:

        def pdf(self, x):
            if x <= -0.5:
                return np.sin(2.0 * np.pi * x) * 0.5 * np.pi
            if x < 0.0:
                return 0.0
            if x <= 0.5:
                return np.sin(2.0 * np.pi * x) * 0.5 * np.pi

        def dpdf(self, x):
            if x <= -0.5:
                return np.cos(2.0 * np.pi * x) * np.pi * np.pi
            if x < 0.0:
                return 0.0
            if x <= 0.5:
                return np.cos(2.0 * np.pi * x) * np.pi * np.pi

        def cdf(self, x):
            if x <= -0.5:
                return 0.25 * (1 - np.cos(2.0 * np.pi * x))
            if x < 0.0:
                return 0.5
            if x <= 0.5:
                return 0.75 - 0.25 * np.cos(2.0 * np.pi * x)

        def support(self):
            return (-1, 0.5)
    dists = [dist0(), dist1()]
    mv0 = [-1 / (2 * np.pi), 1 / 3 - 1 / (4 * np.pi * np.pi)]
    mv1 = [-1 / 4, 3 / 8 - 1 / (2 * np.pi * np.pi) - 1 / 16]
    mvs = [mv0, mv1]

    @pytest.mark.parametrize('dist, mv_ex', zip(dists, mvs))
    @pytest.mark.parametrize('order', [3, 5])
    def test_basic(self, dist, mv_ex, order):
        rng = NumericalInverseHermite(dist, order=order, random_state=42)
        check_cont_samples(rng, dist, mv_ex)

    @pytest.mark.parametrize('domain, err, msg', inf_nan_domains)
    def test_inf_nan_domains(self, domain, err, msg):
        with pytest.raises(err, match=msg):
            NumericalInverseHermite(StandardNormal(), domain=domain)

    def basic_test_all_scipy_dists(self, distname, shapes):
        slow_dists = {'ksone', 'kstwo', 'levy_stable', 'skewnorm'}
        fail_dists = {'beta', 'gausshyper', 'geninvgauss', 'ncf', 'nct', 'norminvgauss', 'genhyperbolic', 'studentized_range', 'vonmises', 'kappa4', 'invgauss', 'wald'}
        if distname in slow_dists:
            pytest.skip('Distribution is too slow')
        if distname in fail_dists:
            pytest.xfail('Fails - usually due to inaccurate CDF/PDF')
        np.random.seed(0)
        dist = getattr(stats, distname)(*shapes)
        fni = NumericalInverseHermite(dist)
        x = np.random.rand(10)
        p_tol = np.max(np.abs(dist.ppf(x) - fni.ppf(x)) / np.abs(dist.ppf(x)))
        u_tol = np.max(np.abs(dist.cdf(fni.ppf(x)) - x))
        assert p_tol < 1e-08
        assert u_tol < 1e-12

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.xslow
    @pytest.mark.parametrize(('distname', 'shapes'), distcont)
    def test_basic_all_scipy_dists(self, distname, shapes):
        self.basic_test_all_scipy_dists(distname, shapes)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_basic_truncnorm_gh17155(self):
        self.basic_test_all_scipy_dists('truncnorm', (0.1, 2))

    def test_input_validation(self):
        match = '`order` must be either 1, 3, or 5.'
        with pytest.raises(ValueError, match=match):
            NumericalInverseHermite(StandardNormal(), order=2)
        match = '`cdf` required but not found'
        with pytest.raises(ValueError, match=match):
            NumericalInverseHermite('norm')
        match = 'could not convert string to float'
        with pytest.raises(ValueError, match=match):
            NumericalInverseHermite(StandardNormal(), u_resolution='ekki')
    rngs = [None, 0, np.random.RandomState(0)]
    rngs.append(np.random.default_rng(0))
    sizes = [(None, tuple()), (8, (8,)), ((4, 5, 6), (4, 5, 6))]

    @pytest.mark.parametrize('rng', rngs)
    @pytest.mark.parametrize('size_in, size_out', sizes)
    def test_RVS(self, rng, size_in, size_out):
        dist = StandardNormal()
        fni = NumericalInverseHermite(dist)
        rng2 = deepcopy(rng)
        rvs = fni.rvs(size=size_in, random_state=rng)
        if size_in is not None:
            assert rvs.shape == size_out
        if rng2 is not None:
            rng2 = check_random_state(rng2)
            uniform = rng2.uniform(size=size_in)
            rvs2 = stats.norm.ppf(uniform)
            assert_allclose(rvs, rvs2)

    def test_inaccurate_CDF(self):
        shapes = (2.3098496451481823, 0.6268795430096368)
        match = '98 : one or more intervals very short; possibly due to numerical problems with a pole or very flat tail'
        with pytest.warns(RuntimeWarning, match=match):
            NumericalInverseHermite(stats.beta(*shapes))
        NumericalInverseHermite(stats.beta(*shapes), u_resolution=1e-08)

    def test_custom_distribution(self):
        dist1 = StandardNormal()
        fni1 = NumericalInverseHermite(dist1)
        dist2 = stats.norm()
        fni2 = NumericalInverseHermite(dist2)
        assert_allclose(fni1.rvs(random_state=0), fni2.rvs(random_state=0))
    u = [np.linspace(0.0, 1.0, num=10000), [], [[]], [np.nan], [-np.inf, np.nan, np.inf], 0, [[np.nan, 0.5, 0.1], [0.2, 0.4, np.inf], [-2, 3, 4]]]

    @pytest.mark.parametrize('u', u)
    def test_ppf(self, u):
        dist = StandardNormal()
        rng = NumericalInverseHermite(dist, u_resolution=1e-12)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in greater')
            sup.filter(RuntimeWarning, 'invalid value encountered in greater_equal')
            sup.filter(RuntimeWarning, 'invalid value encountered in less')
            sup.filter(RuntimeWarning, 'invalid value encountered in less_equal')
            res = rng.ppf(u)
            expected = stats.norm.ppf(u)
        assert_allclose(res, expected, rtol=1e-09, atol=3e-10)
        assert res.shape == expected.shape

    def test_u_error(self):
        dist = StandardNormal()
        rng = NumericalInverseHermite(dist, u_resolution=1e-10)
        max_error, mae = rng.u_error()
        assert max_error < 1e-10
        assert mae <= max_error
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            rng = NumericalInverseHermite(dist, u_resolution=1e-14)
        max_error, mae = rng.u_error()
        assert max_error < 1e-14
        assert mae <= max_error