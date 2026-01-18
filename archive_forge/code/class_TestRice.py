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
class TestRice:

    def test_rice_zero_b(self):
        x = [0.2, 1.0, 5.0]
        assert_(np.isfinite(stats.rice.pdf(x, b=0.0)).all())
        assert_(np.isfinite(stats.rice.logpdf(x, b=0.0)).all())
        assert_(np.isfinite(stats.rice.cdf(x, b=0.0)).all())
        assert_(np.isfinite(stats.rice.logcdf(x, b=0.0)).all())
        q = [0.1, 0.1, 0.5, 0.9]
        assert_(np.isfinite(stats.rice.ppf(q, b=0.0)).all())
        mvsk = stats.rice.stats(0, moments='mvsk')
        assert_(np.isfinite(mvsk).all())
        b = 1e-08
        assert_allclose(stats.rice.pdf(x, 0), stats.rice.pdf(x, b), atol=b, rtol=0)

    def test_rice_rvs(self):
        rvs = stats.rice.rvs
        assert_equal(rvs(b=3.0).size, 1)
        assert_equal(rvs(b=3.0, size=(3, 5)).shape, (3, 5))

    def test_rice_gh9836(self):
        cdf = stats.rice.cdf(np.arange(10, 160, 10), np.arange(10, 160, 10))
        cdf_exp = [0.4800278103504522, 0.4900233218590353, 0.4933500379379548, 0.4950128317658719, 0.4960103776798502, 0.4966753655438764, 0.4971503395812474, 0.4975065620443196, 0.4977836197921638, 0.498005263664955, 0.4981866072661382, 0.4983377260666599, 0.4984655952615694, 0.4985751970541413, 0.4986701850071265]
        assert_allclose(cdf, cdf_exp)
        probabilities = np.arange(0.1, 1, 0.1)
        ppf = stats.rice.ppf(probabilities, 500 / 4, scale=4)
        ppf_exp = [494.8898762347361, 496.649569085835, 497.9184315188069, 499.0026277378915, 500.015999914625, 501.0293721352668, 502.1135684981884, 503.3824312270405, 505.1421247157822]
        assert_allclose(ppf, ppf_exp)
        ppf = scipy.stats.rice.ppf(0.5, np.arange(10, 150, 10))
        ppf_exp = [10.04995862522287, 20.02499480078302, 30.01666512465732, 40.01249934924363, 50.00999966676032, 60.00833314046875, 70.00714273568241, 80.00624991862573, 90.00555549840364, 100.00499995833597, 110.00454542324384, 120.00416664255323, 130.0038461348812, 140.00357141338748]
        assert_allclose(ppf, ppf_exp)