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
class TestMaxwell:

    @pytest.mark.parametrize('x, ref', [(20, 2.2138865931011176e-86), (0.01, 0.9999997340464585)])
    def test_sf(self, x, ref):
        assert_allclose(stats.maxwell.sf(x), ref, rtol=1e-14)

    @pytest.mark.parametrize('q, ref', [(0.001, 4.033142223656157), (0.9999847412109375, 0.0385743284050381), (2 ** (-55), 8.95564974719481)])
    def test_isf(self, q, ref):
        assert_allclose(stats.maxwell.isf(q), ref, rtol=1e-15)