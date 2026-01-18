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
class TestBernoulli:

    def setup_method(self):
        np.random.seed(1234)

    def test_rvs(self):
        vals = stats.bernoulli.rvs(0.75, size=(2, 50))
        assert_(numpy.all(vals >= 0) & numpy.all(vals <= 1))
        assert_(numpy.shape(vals) == (2, 50))
        assert_(vals.dtype.char in typecodes['AllInteger'])
        val = stats.bernoulli.rvs(0.75)
        assert_(isinstance(val, int))
        val = stats.bernoulli(0.75).rvs(3)
        assert_(isinstance(val, numpy.ndarray))
        assert_(val.dtype.char in typecodes['AllInteger'])

    def test_entropy(self):
        b = stats.bernoulli(0.25)
        expected_h = -0.25 * np.log(0.25) - 0.75 * np.log(0.75)
        h = b.entropy()
        assert_allclose(h, expected_h)
        b = stats.bernoulli(0.0)
        h = b.entropy()
        assert_equal(h, 0.0)
        b = stats.bernoulli(1.0)
        h = b.entropy()
        assert_equal(h, 0.0)