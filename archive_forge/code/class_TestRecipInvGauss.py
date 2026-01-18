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
class TestRecipInvGauss:

    def test_pdf_endpoint(self):
        p = stats.recipinvgauss.pdf(0, 0.6)
        assert p == 0.0

    def test_logpdf_endpoint(self):
        logp = stats.recipinvgauss.logpdf(0, 0.6)
        assert logp == -np.inf

    def test_cdf_small_x(self):
        p = stats.recipinvgauss.cdf(0.05, 0.5)
        expected = 6.590396159501331e-20
        assert_allclose(p, expected, rtol=1e-14)

    def test_sf_large_x(self):
        p = stats.recipinvgauss.sf(80, 0.5)
        expected = 2.699819200556787e-18
        assert_allclose(p, expected, 5e-15)