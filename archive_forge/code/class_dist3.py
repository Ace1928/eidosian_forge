import threading
import pickle
import pytest
from copy import deepcopy
import platform
import sys
import math
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy.stats.sampling import (
from pytest import raises as assert_raises
from scipy import stats
from scipy import special
from scipy.stats import chisquare, cramervonmises
from scipy.stats._distr_params import distdiscrete, distcont
from scipy._lib._util import check_random_state
class dist3:

    def pdf(self, x):
        return 0.2 * (0.05 + 0.45 * (1 + np.sin(2 * np.pi * x)))

    def cdf(self, x):
        return x / 10.0 + 0.5 + 0.09 / (2 * np.pi) * (np.cos(10 * np.pi) - np.cos(2 * np.pi * x))

    def support(self):
        return (-5, 5)