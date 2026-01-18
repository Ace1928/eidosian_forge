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
class TestLaplaceasymmetric:

    def test_laplace(self):
        points = np.array([1, 2, 3])
        pdf1 = stats.laplace_asymmetric.pdf(points, 1)
        pdf2 = stats.laplace.pdf(points)
        assert_allclose(pdf1, pdf2)

    def test_asymmetric_laplace_pdf(self):
        points = np.array([1, 2, 3])
        kappa = 2
        kapinv = 1 / kappa
        pdf1 = stats.laplace_asymmetric.pdf(points, kappa)
        pdf2 = stats.laplace_asymmetric.pdf(points * kappa ** 2, kapinv)
        assert_allclose(pdf1, pdf2)

    def test_asymmetric_laplace_log_10_16(self):
        points = np.array([-np.log(16), np.log(10)])
        kappa = 2
        pdf1 = stats.laplace_asymmetric.pdf(points, kappa)
        cdf1 = stats.laplace_asymmetric.cdf(points, kappa)
        sf1 = stats.laplace_asymmetric.sf(points, kappa)
        pdf2 = np.array([1 / 10, 1 / 250])
        cdf2 = np.array([1 / 5, 1 - 1 / 500])
        sf2 = np.array([4 / 5, 1 / 500])
        ppf1 = stats.laplace_asymmetric.ppf(cdf2, kappa)
        ppf2 = points
        isf1 = stats.laplace_asymmetric.isf(sf2, kappa)
        isf2 = points
        assert_allclose(np.concatenate((pdf1, cdf1, sf1, ppf1, isf1)), np.concatenate((pdf2, cdf2, sf2, ppf2, isf2)))