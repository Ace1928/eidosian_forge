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
class TestVonMises:

    @pytest.mark.parametrize('k', [0.1, 1, 101])
    @pytest.mark.parametrize('x', [0, 1, np.pi, 10, 100])
    def test_vonmises_periodic(self, k, x):

        def check_vonmises_pdf_periodic(k, L, s, x):
            vm = stats.vonmises(k, loc=L, scale=s)
            assert_almost_equal(vm.pdf(x), vm.pdf(x % (2 * np.pi * s)))

        def check_vonmises_cdf_periodic(k, L, s, x):
            vm = stats.vonmises(k, loc=L, scale=s)
            assert_almost_equal(vm.cdf(x) % 1, vm.cdf(x % (2 * np.pi * s)) % 1)
        check_vonmises_pdf_periodic(k, 0, 1, x)
        check_vonmises_pdf_periodic(k, 1, 1, x)
        check_vonmises_pdf_periodic(k, 0, 10, x)
        check_vonmises_cdf_periodic(k, 0, 1, x)
        check_vonmises_cdf_periodic(k, 1, 1, x)
        check_vonmises_cdf_periodic(k, 0, 10, x)

    def test_vonmises_line_support(self):
        assert_equal(stats.vonmises_line.a, -np.pi)
        assert_equal(stats.vonmises_line.b, np.pi)

    def test_vonmises_numerical(self):
        vm = stats.vonmises(800)
        assert_almost_equal(vm.cdf(0), 0.5)

    @pytest.mark.parametrize('x, kappa, expected_pdf', [(0.1, 0.01, 0.16074242744907072), (0.1, 25.0, 1.7515464099118245), (0.1, 800, 0.2073272544458798), (2.0, 0.01, 0.15849003875385817), (2.0, 25.0, 8.356882934278192e-16), (2.0, 800, 0.0)])
    def test_vonmises_pdf(self, x, kappa, expected_pdf):
        pdf = stats.vonmises.pdf(x, kappa)
        assert_allclose(pdf, expected_pdf, rtol=1e-15)

    @pytest.mark.parametrize('kappa, expected_entropy', [(1, 1.6274014590199897), (5, 0.6756431570114528), (100, -0.8811275441649473), (1000, -2.03468891852547), (2000, -2.3813876496587847)])
    def test_vonmises_entropy(self, kappa, expected_entropy):
        entropy = stats.vonmises.entropy(kappa)
        assert_allclose(entropy, expected_entropy, rtol=1e-13)

    def test_vonmises_rvs_gh4598(self):
        seed = abs(hash('von_mises_rvs'))
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        rng3 = np.random.default_rng(seed)
        rvs1 = stats.vonmises(1, loc=0, scale=1).rvs(random_state=rng1)
        rvs2 = stats.vonmises(1, loc=2 * np.pi, scale=1).rvs(random_state=rng2)
        rvs3 = stats.vonmises(1, loc=0, scale=2 * np.pi / abs(rvs1) + 1).rvs(random_state=rng3)
        assert_allclose(rvs1, rvs2, atol=1e-15)
        assert_allclose(rvs1, rvs3, atol=1e-15)

    @pytest.mark.parametrize('x, kappa, expected_logpdf', [(0.1, 0.01, -1.827952024600317), (0.1, 25.0, 0.5604990605420549), (0.1, 800, -1.5734567947337514), (2.0, 0.01, -1.8420635346185685), (2.0, 25.0, -34.718275985087146), (2.0, 800, -1130.4942582548683)])
    def test_vonmises_logpdf(self, x, kappa, expected_logpdf):
        logpdf = stats.vonmises.logpdf(x, kappa)
        assert_allclose(logpdf, expected_logpdf, rtol=1e-15)

    def test_vonmises_expect(self):
        """
        Test that the vonmises expectation values are
        computed correctly.  This test checks that the
        numeric integration estimates the correct normalization
        (1) and mean angle (loc).  These expectations are
        independent of the chosen 2pi interval.
        """
        rng = np.random.default_rng(6762668991392531563)
        loc, kappa, lb = rng.random(3) * 10
        res = stats.vonmises(loc=loc, kappa=kappa).expect(lambda x: 1)
        assert_allclose(res, 1)
        assert np.issubdtype(res.dtype, np.floating)
        bounds = (lb, lb + 2 * np.pi)
        res = stats.vonmises(loc=loc, kappa=kappa).expect(lambda x: 1, *bounds)
        assert_allclose(res, 1)
        assert np.issubdtype(res.dtype, np.floating)
        bounds = (lb, lb + 2 * np.pi)
        res = stats.vonmises(loc=loc, kappa=kappa).expect(lambda x: np.exp(1j * x), *bounds, complex_func=1)
        assert_allclose(np.angle(res), loc % (2 * np.pi))
        assert np.issubdtype(res.dtype, np.complexfloating)

    @pytest.mark.xslow
    @pytest.mark.parametrize('rvs_loc', [0, 2])
    @pytest.mark.parametrize('rvs_shape', [1, 100, 100000000.0])
    @pytest.mark.parametrize('fix_loc', [True, False])
    @pytest.mark.parametrize('fix_shape', [True, False])
    def test_fit_MLE_comp_optimizer(self, rvs_loc, rvs_shape, fix_loc, fix_shape):
        if fix_shape and fix_loc:
            pytest.skip('Nothing to fit.')
        rng = np.random.default_rng(6762668991392531563)
        data = stats.vonmises.rvs(rvs_shape, size=1000, loc=rvs_loc, random_state=rng)
        kwds = {'fscale': 1}
        if fix_loc:
            kwds['floc'] = rvs_loc
        if fix_shape:
            kwds['f0'] = rvs_shape
        _assert_less_or_close_loglike(stats.vonmises, data, stats.vonmises.nnlf, **kwds)

    @pytest.mark.xslow
    def test_vonmises_fit_bad_floc(self):
        data = [-0.92923506, -0.32498224, 0.13054989, -0.97252014, 2.79658071, -0.89110948, 1.22520295, 1.44398065, 2.49163859, 1.50315096, 3.05437696, -2.73126329, -3.06272048, 1.64647173, 1.94509247, -1.14328023, 0.8499056, 2.36714682, -1.6823179, -0.88359996]
        data = np.asarray(data)
        loc = -0.5 * np.pi
        kappa_fit, loc_fit, scale_fit = stats.vonmises.fit(data, floc=loc)
        assert kappa_fit == np.finfo(float).tiny
        _assert_less_or_close_loglike(stats.vonmises, data, stats.vonmises.nnlf, fscale=1, floc=loc)

    @pytest.mark.parametrize('sign', [-1, 1])
    def test_vonmises_fit_unwrapped_data(self, sign):
        rng = np.random.default_rng(6762668991392531563)
        data = stats.vonmises(loc=sign * 0.5 * np.pi, kappa=10).rvs(100000, random_state=rng)
        shifted_data = data + 4 * np.pi
        kappa_fit, loc_fit, scale_fit = stats.vonmises.fit(data)
        kappa_fit_shifted, loc_fit_shifted, _ = stats.vonmises.fit(shifted_data)
        assert_allclose(loc_fit, loc_fit_shifted)
        assert_allclose(kappa_fit, kappa_fit_shifted)
        assert scale_fit == 1
        assert -np.pi < loc_fit < np.pi