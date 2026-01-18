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
class TestInvGamma:

    def test_invgamma_inf_gh_1866(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            mvsk = stats.invgamma.stats(a=19.31, moments='mvsk')
            expected = [0.0546149645, 0.0001723162534, 1.020362676, 2.055616582]
            assert_allclose(mvsk, expected)
            a = [1.1, 3.1, 5.6]
            mvsk = stats.invgamma.stats(a=a, moments='mvsk')
            expected = ([10.0, 0.476190476, 0.2173913043], [np.inf, 0.2061430632, 0.01312749422], [np.nan, 41.95235392, 2.919025532], [np.nan, np.nan, 24.51923076])
            for x, y in zip(mvsk, expected):
                assert_almost_equal(x, y)

    def test_cdf_ppf(self):
        x = np.logspace(-2.6, 0)
        y = stats.invgamma.cdf(x, 1)
        xx = stats.invgamma.ppf(y, 1)
        assert_allclose(x, xx)

    def test_sf_isf(self):
        if sys.maxsize > 2 ** 32:
            x = np.logspace(2, 100)
        else:
            x = np.logspace(2, 18)
        y = stats.invgamma.sf(x, 1)
        xx = stats.invgamma.isf(y, 1)
        assert_allclose(x, xx, rtol=1.0)

    @pytest.mark.parametrize('a, ref', [(100000000.0, -26.21208257605721), (1e+100, -343.9688254159022)])
    def test_large_entropy(self, a, ref):
        assert_allclose(stats.invgamma.entropy(a), ref, rtol=1e-15)