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
class TestNBinom:

    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.nbinom.rvs(10, 0.75, size=(2, 50))
        assert_(numpy.all(vals >= 0))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.nbinom.rvs(10, 0.75)
        assert_(isinstance(val, int))
        val = stats.nbinom(10, 0.75).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_pmf(self):
        assert_allclose(np.exp(stats.nbinom.logpmf(700, 721, 0.52)), stats.nbinom.pmf(700, 721, 0.52))
        val = scipy.stats.nbinom.logpmf(0, 1, 1)
        assert_equal(val, 0)

    def test_logcdf_gh16159(self):
        vals = stats.nbinom.logcdf([0, 5, 0, 5], n=4.8, p=0.45)
        ref = np.log(stats.nbinom.cdf([0, 5, 0, 5], n=4.8, p=0.45))
        assert_allclose(vals, ref)