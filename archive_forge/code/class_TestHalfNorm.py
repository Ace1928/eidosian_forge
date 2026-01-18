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
class TestHalfNorm:

    @pytest.mark.parametrize('x, sfx', [(1, 0.3173105078629141), (10, 1.523970604832105e-23)])
    def test_sf_isf(self, x, sfx):
        assert_allclose(stats.halfnorm.sf(x), sfx, rtol=1e-14)
        assert_allclose(stats.halfnorm.isf(sfx), x, rtol=1e-14)

    @pytest.mark.parametrize('x, ref', [(1e-40, 7.978845608028653e-41), (1e-18, 7.978845608028654e-19), (8, 0.9999999999999988)])
    def test_cdf(self, x, ref):
        assert_allclose(stats.halfnorm.cdf(x), ref, rtol=1e-15)

    @pytest.mark.parametrize('rvs_loc', [1e-05, 10000000000.0])
    @pytest.mark.parametrize('rvs_scale', [0.01, 100, 100000000.0])
    @pytest.mark.parametrize('fix_loc', [True, False])
    @pytest.mark.parametrize('fix_scale', [True, False])
    def test_fit_MLE_comp_optimizer(self, rvs_loc, rvs_scale, fix_loc, fix_scale):
        rng = np.random.default_rng(6762668991392531563)
        data = stats.halfnorm.rvs(loc=rvs_loc, scale=rvs_scale, size=1000, random_state=rng)
        if fix_loc and fix_scale:
            error_msg = 'All parameters fixed. There is nothing to optimize.'
            with pytest.raises(RuntimeError, match=error_msg):
                stats.halflogistic.fit(data, floc=rvs_loc, fscale=rvs_scale)
            return
        kwds = {}
        if fix_loc:
            kwds['floc'] = rvs_loc
        if fix_scale:
            kwds['fscale'] = rvs_scale
        _assert_less_or_close_loglike(stats.halfnorm, data, **kwds)

    def test_fit_error(self):
        with pytest.raises(FitDataError):
            stats.halfnorm.fit([1, 2, 3], floc=2)