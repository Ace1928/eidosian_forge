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
class TestGamma:

    def test_pdf(self):
        pdf = stats.gamma.pdf(90, 394, scale=1.0 / 5)
        assert_almost_equal(pdf, 0.002312341)
        pdf = stats.gamma.pdf(3, 10, scale=1.0 / 5)
        assert_almost_equal(pdf, 0.1620358)

    def test_logpdf(self):
        logpdf = stats.gamma.logpdf(0, 1)
        assert_almost_equal(logpdf, 0)

    def test_fit_bad_keyword_args(self):
        x = [0.1, 0.5, 0.6]
        assert_raises(TypeError, stats.gamma.fit, x, floc=0, plate='shrimp')

    def test_isf(self):
        assert np.isclose(stats.gamma.isf(1e-17, 1), 39.14394658089878, atol=1e-14)
        assert np.isclose(stats.gamma.isf(1e-50, 100), 330.6557590436547, atol=1e-13)

    @pytest.mark.parametrize('scale', [1.0, 5.0])
    def test_delta_cdf(self, scale):
        delta = stats.gamma._delta_cdf(scale * 245, scale * 250, 3, scale=scale)
        assert_allclose(delta, 1.1902609356171962e-102, rtol=1e-13)

    @pytest.mark.parametrize('a, ref, rtol', [(0.0001, -9990.366610819761, 1e-15), (2, 1.5772156649015328, 1e-15), (100, 3.7181819485047463, 1e-13), (10000.0, 6.024075385026086, 1e-15), (1e+18, 22.142204370151084, 1e-15), (1e+100, 116.54819318290696, 1e-15)])
    def test_entropy(self, a, ref, rtol):
        assert_allclose(stats.gamma.entropy(a), ref, rtol=rtol)