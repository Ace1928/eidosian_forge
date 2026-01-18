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
class TestPareto:

    def test_stats(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            m, v, s, k = stats.pareto.stats(0.5, moments='mvsk')
            assert_equal(m, np.inf)
            assert_equal(v, np.inf)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)
            m, v, s, k = stats.pareto.stats(1.0, moments='mvsk')
            assert_equal(m, np.inf)
            assert_equal(v, np.inf)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)
            m, v, s, k = stats.pareto.stats(1.5, moments='mvsk')
            assert_equal(m, 3.0)
            assert_equal(v, np.inf)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)
            m, v, s, k = stats.pareto.stats(2.0, moments='mvsk')
            assert_equal(m, 2.0)
            assert_equal(v, np.inf)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)
            m, v, s, k = stats.pareto.stats(2.5, moments='mvsk')
            assert_allclose(m, 2.5 / 1.5)
            assert_allclose(v, 2.5 / (1.5 * 1.5 * 0.5))
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)
            m, v, s, k = stats.pareto.stats(3.0, moments='mvsk')
            assert_allclose(m, 1.5)
            assert_allclose(v, 0.75)
            assert_equal(s, np.nan)
            assert_equal(k, np.nan)
            m, v, s, k = stats.pareto.stats(3.5, moments='mvsk')
            assert_allclose(m, 3.5 / 2.5)
            assert_allclose(v, 3.5 / (2.5 * 2.5 * 1.5))
            assert_allclose(s, 2 * 4.5 / 0.5 * np.sqrt(1.5 / 3.5))
            assert_equal(k, np.nan)
            m, v, s, k = stats.pareto.stats(4.0, moments='mvsk')
            assert_allclose(m, 4.0 / 3.0)
            assert_allclose(v, 4.0 / 18.0)
            assert_allclose(s, 2 * (1 + 4.0) / (4.0 - 3) * np.sqrt((4.0 - 2) / 4.0))
            assert_equal(k, np.nan)
            m, v, s, k = stats.pareto.stats(4.5, moments='mvsk')
            assert_allclose(m, 4.5 / 3.5)
            assert_allclose(v, 4.5 / (3.5 * 3.5 * 2.5))
            assert_allclose(s, 2 * 5.5 / 1.5 * np.sqrt(2.5 / 4.5))
            assert_allclose(k, 6 * (4.5 ** 3 + 4.5 ** 2 - 6 * 4.5 - 2) / (4.5 * 1.5 * 0.5))

    def test_sf(self):
        x = 1000000000.0
        b = 2
        scale = 1.5
        p = stats.pareto.sf(x, b, loc=0, scale=scale)
        expected = (scale / x) ** b
        assert_allclose(p, expected)

    @pytest.fixture(scope='function')
    def rng(self):
        return np.random.default_rng(1234)

    @pytest.mark.filterwarnings('ignore:invalid value encountered in double_scalars')
    @pytest.mark.parametrize('rvs_shape', [1, 2])
    @pytest.mark.parametrize('rvs_loc', [0, 2])
    @pytest.mark.parametrize('rvs_scale', [1, 5])
    def test_fit(self, rvs_shape, rvs_loc, rvs_scale, rng):
        data = stats.pareto.rvs(size=100, b=rvs_shape, scale=rvs_scale, loc=rvs_loc, random_state=rng)
        shape_mle_analytical1 = stats.pareto.fit(data, floc=0, f0=1.04)[0]
        shape_mle_analytical2 = stats.pareto.fit(data, floc=0, fix_b=1.04)[0]
        shape_mle_analytical3 = stats.pareto.fit(data, floc=0, fb=1.04)[0]
        assert shape_mle_analytical1 == shape_mle_analytical2 == shape_mle_analytical3 == 1.04
        data = stats.pareto.rvs(size=100, b=rvs_shape, scale=rvs_scale, loc=rvs_loc + 2, random_state=rng)
        shape_mle_a, loc_mle_a, scale_mle_a = stats.pareto.fit(data, floc=2)
        assert_equal(scale_mle_a + 2, data.min())
        data_shift = data - 2
        ndata = data_shift.shape[0]
        assert_equal(shape_mle_a, ndata / np.sum(np.log(data_shift / data_shift.min())))
        assert_equal(loc_mle_a, 2)

    @pytest.mark.parametrize('rvs_shape', [0.1, 2])
    @pytest.mark.parametrize('rvs_loc', [0, 2])
    @pytest.mark.parametrize('rvs_scale', [1, 5])
    @pytest.mark.parametrize('fix_shape, fix_loc, fix_scale', [p for p in product([True, False], repeat=3) if False in p])
    @np.errstate(invalid='ignore')
    def test_fit_MLE_comp_optimizer(self, rvs_shape, rvs_loc, rvs_scale, fix_shape, fix_loc, fix_scale, rng):
        data = stats.pareto.rvs(size=100, b=rvs_shape, scale=rvs_scale, loc=rvs_loc, random_state=rng)
        kwds = {}
        if fix_shape:
            kwds['f0'] = rvs_shape
        if fix_loc:
            kwds['floc'] = rvs_loc
        if fix_scale:
            kwds['fscale'] = rvs_scale
        _assert_less_or_close_loglike(stats.pareto, data, **kwds)

    @np.errstate(invalid='ignore')
    def test_fit_known_bad_seed(self):
        shape, location, scale = (1, 0, 1)
        data = stats.pareto.rvs(shape, location, scale, size=100, random_state=np.random.default_rng(2535619))
        _assert_less_or_close_loglike(stats.pareto, data)

    def test_fit_warnings(self):
        assert_fit_warnings(stats.pareto)
        assert_raises(FitDataError, stats.pareto.fit, [1, 2, 3], floc=2)
        assert_raises(FitDataError, stats.pareto.fit, [5, 2, 3], floc=1, fscale=3)

    def test_negative_data(self, rng):
        data = stats.pareto.rvs(loc=-130, b=1, size=100, random_state=rng)
        assert_array_less(data, 0)
        _ = stats.pareto.fit(data)