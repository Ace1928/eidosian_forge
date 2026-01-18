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
class TestHypSecant:

    @pytest.mark.parametrize('x, reference', [(30, 5.957247804324683e-14), (50, 1.2278802891647964e-22)])
    def test_sf(self, x, reference):
        sf = stats.hypsecant.sf(x)
        assert_allclose(sf, reference, rtol=5e-15)

    @pytest.mark.parametrize('p, reference', [(1e-06, 13.363927852673998), (1e-12, 27.179438410639094)])
    def test_isf(self, p, reference):
        x = stats.hypsecant.isf(p)
        assert_allclose(x, reference, rtol=5e-15)