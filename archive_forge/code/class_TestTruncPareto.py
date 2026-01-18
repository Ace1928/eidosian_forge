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
class TestTruncPareto:

    def test_pdf(self):
        b, c = (1.8, 5.3)
        x = np.linspace(1.8, 5.3)
        res = stats.truncpareto(b, c).pdf(x)
        ref = stats.pareto(b).pdf(x) / stats.pareto(b).cdf(c)
        assert_allclose(res, ref)

    @pytest.mark.parametrize('fix_loc', [True, False])
    @pytest.mark.parametrize('fix_scale', [True, False])
    @pytest.mark.parametrize('fix_b', [True, False])
    @pytest.mark.parametrize('fix_c', [True, False])
    def test_fit(self, fix_loc, fix_scale, fix_b, fix_c):
        rng = np.random.default_rng(6747363148258237171)
        b, c, loc, scale = (1.8, 5.3, 1, 2.5)
        dist = stats.truncpareto(b, c, loc=loc, scale=scale)
        data = dist.rvs(size=500, random_state=rng)
        kwds = {}
        if fix_loc:
            kwds['floc'] = loc
        if fix_scale:
            kwds['fscale'] = scale
        if fix_b:
            kwds['f0'] = b
        if fix_c:
            kwds['f1'] = c
        if fix_loc and fix_scale and fix_b and fix_c:
            message = 'All parameters fixed. There is nothing to optimize.'
            with pytest.raises(RuntimeError, match=message):
                stats.truncpareto.fit(data, **kwds)
        else:
            _assert_less_or_close_loglike(stats.truncpareto, data, **kwds)