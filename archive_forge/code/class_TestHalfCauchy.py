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
class TestHalfCauchy:

    @pytest.mark.parametrize('rvs_loc', [1e-05, 10000000000.0])
    @pytest.mark.parametrize('rvs_scale', [0.01, 100000000.0])
    @pytest.mark.parametrize('fix_loc', [True, False])
    @pytest.mark.parametrize('fix_scale', [True, False])
    def test_fit_MLE_comp_optimizer(self, rvs_loc, rvs_scale, fix_loc, fix_scale):
        rng = np.random.default_rng(6762668991392531563)
        data = stats.halfnorm.rvs(loc=rvs_loc, scale=rvs_scale, size=1000, random_state=rng)
        if fix_loc and fix_scale:
            error_msg = 'All parameters fixed. There is nothing to optimize.'
            with pytest.raises(RuntimeError, match=error_msg):
                stats.halfcauchy.fit(data, floc=rvs_loc, fscale=rvs_scale)
            return
        kwds = {}
        if fix_loc:
            kwds['floc'] = rvs_loc
        if fix_scale:
            kwds['fscale'] = rvs_scale
        _assert_less_or_close_loglike(stats.halfcauchy, data, **kwds)

    def test_fit_error(self):
        with pytest.raises(FitDataError):
            stats.halfcauchy.fit([1, 2, 3], floc=2)