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
class TestRdist:

    def test_rdist_cdf_gh1285(self):
        distfn = stats.rdist
        values = [0.001, 0.5, 0.999]
        assert_almost_equal(distfn.cdf(distfn.ppf(values, 541.0), 541.0), values, decimal=5)

    def test_rdist_beta(self):
        x = np.linspace(-0.99, 0.99, 10)
        c = 2.7
        assert_almost_equal(0.5 * stats.beta(c / 2, c / 2).pdf((x + 1) / 2), stats.rdist(c).pdf(x))

    @pytest.mark.parametrize('x, c, ref', [(0.0001, 541, 0.49907251345565845), (0.1, 241, 0.06000788166249205), (0.5, 441, 1.0655898106047832e-29), (0.8, 341, 6.025478373732215e-78)])
    def test_rdist_sf(self, x, c, ref):
        assert_allclose(stats.rdist.sf(x, c), ref, rtol=5e-14)