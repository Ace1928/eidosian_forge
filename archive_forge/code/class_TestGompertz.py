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
class TestGompertz:

    def test_gompertz_accuracy(self):
        p = stats.gompertz.ppf(stats.gompertz.cdf(1e-100, 1), 1)
        assert_allclose(p, 1e-100)

    @pytest.mark.parametrize('x, c, sfx', [(1, 2.5, 0.013626967146253437), (3, 2.5, 1.8973243273704087e-21), (0.05, 5, 0.7738668242570479), (2.25, 5, 3.707795833465481e-19)])
    def test_sf_isf(self, x, c, sfx):
        assert_allclose(stats.gompertz.sf(x, c), sfx, rtol=1e-14)
        assert_allclose(stats.gompertz.isf(sfx, c), x, rtol=1e-14)

    @pytest.mark.parametrize('c, ref', [(0.0001, 1.5762523017634573), (1, 0.4036526376768059), (1000, -5.908754280976161), (10000000000.0, -22.025850930040455)])
    def test_entropy(self, c, ref):
        assert_allclose(stats.gompertz.entropy(c), ref, rtol=1e-14)