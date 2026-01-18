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
class TestChi2:

    def test_precision(self):
        assert_almost_equal(stats.chi2.pdf(1000, 1000), 0.008919133934753128, decimal=14)
        assert_almost_equal(stats.chi2.pdf(100, 100), 0.028162503162596778, decimal=14)

    def test_ppf(self):
        df = 4.8
        x = stats.chi2.ppf(2e-47, df)
        assert_allclose(x, 1.0984724795751798e-19, rtol=1e-10)
        x = stats.chi2.ppf(0.5, df)
        assert_allclose(x, 4.152314075985894, rtol=1e-10)
        df = 13
        x = stats.chi2.ppf(2e-77, df)
        assert_allclose(x, 1.0106330688195198e-11, rtol=1e-10)
        x = stats.chi2.ppf(0.1, df)
        assert_allclose(x, 7.041504580095462, rtol=1e-10)

    @pytest.mark.parametrize('df, ref', [(0.0001, -19988.980448690163), (1, 0.7837571104739337), (100, 4.061397128938114), (251, 4.525577254045129), (1000000000000000.0, 19.034900320939986)])
    def test_entropy(self, df, ref):
        assert_allclose(stats.chi2(df).entropy(), ref, rtol=1e-13)