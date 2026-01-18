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
class TestPowerNorm:

    @pytest.mark.parametrize('x, c, ref', [(9, 1, 1.1285884059538405e-19), (20, 2, 7.582445786569958e-178), (100, 0.02, 3.330957891903866e-44), (200, 0.01, 1.3004759092324774e-87)])
    def test_sf(self, x, c, ref):
        assert_allclose(stats.powernorm.sf(x, c), ref, rtol=1e-13)

    @pytest.mark.parametrize('q, c, ref', [(1e-05, 20, -0.15690800666514138), (0.99999, 100, -5.19933666203545), (0.9999, 0.02, -2.576676052143387), (0.05, 0.02, 17.089518110222244), (1e-18, 2, 5.9978070150076865), (1e-50, 5, 6.361340902404057)])
    def test_isf(self, q, c, ref):
        assert_allclose(stats.powernorm.isf(q, c), ref, rtol=5e-12)

    @pytest.mark.parametrize('x, c, ref', [(-12, 9, 1.598833900869911e-32), (2, 9, 0.9999999999999983), (-20, 9, 2.4782617067456103e-88), (-5, 0.02, 5.733032242841443e-09), (-20, 0.02, 5.507248237212467e-91)])
    def test_cdf(self, x, c, ref):
        assert_allclose(stats.powernorm.cdf(x, c), ref, rtol=5e-14)