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
class StandardNormal:

    def pdf(self, x):
        return 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * x * x)

    def dpdf(self, x):
        return 1.0 / np.sqrt(2.0 * np.pi) * -x * np.exp(-0.5 * x * x)

    def cdf(self, x):
        return special.ndtr(x)