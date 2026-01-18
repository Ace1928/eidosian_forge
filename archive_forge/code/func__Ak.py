from collections import namedtuple
from dataclasses import dataclass
from math import comb
import numpy as np
import warnings
from itertools import combinations
import scipy.stats
from scipy.optimize import shgo
from . import distributions
from ._common import ConfidenceInterval
from ._continuous_distns import chi2, norm
from scipy.special import gamma, kv, gammaln
from scipy.fft import ifft
from ._stats_pythran import _a_ij_Aij_Dij2
from ._stats_pythran import (
from ._axis_nan_policy import _axis_nan_policy_factory
from scipy.stats import _stats_py
def _Ak(k, x):
    m = 2 * k + 1
    sx = 2 * np.sqrt(x)
    y1 = x ** (3 / 4)
    y2 = x ** (5 / 4)
    e1 = m * gamma(k + 1 / 2) * _ed2((4 * k + 3) / sx) / (9 * y1)
    e2 = gamma(k + 1 / 2) * _ed3((4 * k + 1) / sx) / (72 * y2)
    e3 = 2 * (m + 2) * gamma(k + 3 / 2) * _ed3((4 * k + 5) / sx) / (12 * y2)
    e4 = 7 * m * gamma(k + 1 / 2) * _ed2((4 * k + 1) / sx) / (144 * y1)
    e5 = 7 * m * gamma(k + 1 / 2) * _ed2((4 * k + 5) / sx) / (144 * y1)
    return e1 + e2 + e3 + e4 + e5