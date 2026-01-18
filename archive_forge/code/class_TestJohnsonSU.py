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
class TestJohnsonSU:

    @pytest.mark.parametrize('case', [(-0.01, 1.1, 0.02, 0.0001, 0.02000137427557091, 2.1112742956578063e-08, 0.05989781342460999, 20.36324408592951 - 3), (2.554395574161155, 2.2482281679651965, 0, 1, -1.54215386737391, 0.7629882028469993, -1.256656139406788, 6.303058419339775 - 3)])
    def test_moment_gh18071(self, case):
        res = stats.johnsonsu.stats(*case[:4], moments='mvsk')
        assert_allclose(res, case[4:], rtol=1e-14)