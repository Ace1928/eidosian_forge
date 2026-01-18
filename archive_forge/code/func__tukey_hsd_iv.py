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
def _tukey_hsd_iv(args):
    if len(args) < 2:
        raise ValueError('There must be more than 1 treatment.')
    args = [np.asarray(arg) for arg in args]
    for arg in args:
        if arg.ndim != 1:
            raise ValueError('Input samples must be one-dimensional.')
        if arg.size <= 1:
            raise ValueError('Input sample size must be greater than one.')
        if np.isinf(arg).any():
            raise ValueError('Input samples must be finite.')
    return args