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
class dist2:

    def pdf(self, x):
        return 0.05 + 0.45 * (1 + np.sin(2 * np.pi * x))

    def cdf(self, x):
        return 0.05 * (x + 1) + 0.9 * (1.0 + 2.0 * np.pi * (1 + x) - np.cos(2.0 * np.pi * x)) / (4.0 * np.pi)

    def support(self):
        return (-1, 1)