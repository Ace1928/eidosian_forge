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
class dist1:

    def pdf(self, x):
        if x <= -0.5:
            return np.sin(2.0 * np.pi * x) * 0.5 * np.pi
        if x < 0.0:
            return 0.0
        if x <= 0.5:
            return np.sin(2.0 * np.pi * x) * 0.5 * np.pi

    def dpdf(self, x):
        if x <= -0.5:
            return np.cos(2.0 * np.pi * x) * np.pi * np.pi
        if x < 0.0:
            return 0.0
        if x <= 0.5:
            return np.cos(2.0 * np.pi * x) * np.pi * np.pi

    def cdf(self, x):
        if x <= -0.5:
            return 0.25 * (1 - np.cos(2.0 * np.pi * x))
        if x < 0.0:
            return 0.5
        if x <= 0.5:
            return 0.75 - 0.25 * np.cos(2.0 * np.pi * x)

    def support(self):
        return (-1, 0.5)