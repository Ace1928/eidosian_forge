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
class TestGenLogistic:

    @pytest.mark.parametrize('x, expected', [(-1000, -1499.5945348918917), (-125, -187.09453489189184), (0, -1.3274028432916989), (100, -99.59453489189184), (1000, -999.5945348918918)])
    def test_logpdf(self, x, expected):
        c = 1.5
        logp = stats.genlogistic.logpdf(x, c)
        assert_allclose(logp, expected, rtol=1e-13)

    @pytest.mark.parametrize('c, ref', [(1e-100, 231.25850929940458), (0.0001, 10.21050485336338), (100000000.0, 1.577215669901533), (1e+100, 1.5772156649015328)])
    def test_entropy(self, c, ref):
        assert_allclose(stats.genlogistic.entropy(c), ref, rtol=5e-15)

    @pytest.mark.parametrize('x, c, ref', [(200, 10, 1.3838965267367375e-86), (500, 20, 1.424915281348257e-216)])
    def test_sf(self, x, c, ref):
        assert_allclose(stats.genlogistic.sf(x, c), ref, rtol=1e-14)

    @pytest.mark.parametrize('q, c, ref', [(0.01, 200, 9.898441467379765), (0.001, 2, 7.600152115573173)])
    def test_isf(self, q, c, ref):
        assert_allclose(stats.genlogistic.isf(q, c), ref, rtol=5e-16)

    @pytest.mark.parametrize('q, c, ref', [(0.5, 200, 5.6630969187064615), (0.99, 20, 7.595630231412436)])
    def test_ppf(self, q, c, ref):
        assert_allclose(stats.genlogistic.ppf(q, c), ref, rtol=5e-16)

    @pytest.mark.parametrize('x, c, ref', [(100, 0.02, -7.440151952041672e-46), (50, 20, -3.857499695927835e-21)])
    def test_logcdf(self, x, c, ref):
        assert_allclose(stats.genlogistic.logcdf(x, c), ref, rtol=1e-15)