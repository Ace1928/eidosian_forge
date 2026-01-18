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
class TestKappa4:

    def test_cdf_genpareto(self):
        x = [0.0, 0.1, 0.2, 0.5]
        h = 1.0
        for k in [-1.9, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 1.9]:
            vals = stats.kappa4.cdf(x, h, k)
            vals_comp = stats.genpareto.cdf(x, -k)
            assert_allclose(vals, vals_comp)

    def test_cdf_genextreme(self):
        x = np.linspace(-5, 5, 10)
        h = 0.0
        k = np.linspace(-3, 3, 10)
        vals = stats.kappa4.cdf(x, h, k)
        vals_comp = stats.genextreme.cdf(x, k)
        assert_allclose(vals, vals_comp)

    def test_cdf_expon(self):
        x = np.linspace(0, 10, 10)
        h = 1.0
        k = 0.0
        vals = stats.kappa4.cdf(x, h, k)
        vals_comp = stats.expon.cdf(x)
        assert_allclose(vals, vals_comp)

    def test_cdf_gumbel_r(self):
        x = np.linspace(-5, 5, 10)
        h = 0.0
        k = 0.0
        vals = stats.kappa4.cdf(x, h, k)
        vals_comp = stats.gumbel_r.cdf(x)
        assert_allclose(vals, vals_comp)

    def test_cdf_logistic(self):
        x = np.linspace(-5, 5, 10)
        h = -1.0
        k = 0.0
        vals = stats.kappa4.cdf(x, h, k)
        vals_comp = stats.logistic.cdf(x)
        assert_allclose(vals, vals_comp)

    def test_cdf_uniform(self):
        x = np.linspace(-5, 5, 10)
        h = 1.0
        k = 1.0
        vals = stats.kappa4.cdf(x, h, k)
        vals_comp = stats.uniform.cdf(x)
        assert_allclose(vals, vals_comp)

    def test_integers_ctor(self):
        stats.kappa4(1, 2)