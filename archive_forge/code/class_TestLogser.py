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
class TestLogser:

    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.logser.rvs(0.75, size=(2, 50))
        assert_(numpy.all(vals >= 1))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.logser.rvs(0.75)
        assert_(isinstance(val, int))
        val = stats.logser(0.75).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_pmf_small_p(self):
        m = stats.logser.pmf(4, 1e-20)
        assert_allclose(m, 2.5e-61)

    def test_mean_small_p(self):
        m = stats.logser.mean(1e-08)
        assert_allclose(m, 1.000000005)