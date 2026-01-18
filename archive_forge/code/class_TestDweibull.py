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
class TestDweibull:

    def test_entropy(self):
        rng = np.random.default_rng(8486259129157041777)
        c = 10 ** rng.normal(scale=100, size=10)
        res = stats.dweibull.entropy(c)
        ref = stats.weibull_min.entropy(c) - np.log(0.5)
        assert_allclose(res, ref, rtol=1e-15)

    def test_sf(self):
        rng = np.random.default_rng(8486259129157041777)
        c = 10 ** rng.normal(scale=1, size=10)
        x = 10 * rng.uniform()
        res = stats.dweibull.sf(x, c)
        ref = 0.5 * stats.weibull_min.sf(x, c)
        assert_allclose(res, ref, rtol=1e-15)