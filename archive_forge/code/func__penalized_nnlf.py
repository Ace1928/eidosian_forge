from scipy._lib._util import getfullargspec_no_self as _getfullargspec
import sys
import keyword
import re
import types
import warnings
from itertools import zip_longest
from scipy._lib import doccer
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state
from scipy.special import comb, entr
from scipy import optimize
from scipy import integrate
from scipy._lib._finite_differences import _derivative
from scipy import stats
from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
import numpy as np
from ._constants import _XMAX, _LOGXMAX
from ._censored_data import CensoredData
from scipy.stats._warnings_errors import FitError
def _penalized_nnlf(self, theta, x):
    """Penalized negative loglikelihood function.

        i.e., - sum (log pdf(x, theta), axis=0) + penalty
        where theta are the parameters (including loc and scale)
        """
    loc, scale, args = self._unpack_loc_scale(theta)
    if not self._argcheck(*args) or scale <= 0:
        return inf
    if isinstance(x, CensoredData):
        x = (x - loc) / scale
        n_log_scale = (len(x) - x.num_censored()) * log(scale)
    else:
        x = (x - loc) / scale
        n_log_scale = len(x) * log(scale)
    return self._nnlf_and_penalty(x, args) + n_log_scale