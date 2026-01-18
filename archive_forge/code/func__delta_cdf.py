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
def _delta_cdf(self, x1, x2, *args, loc=0, scale=1):
    """
        Compute CDF(x2) - CDF(x1).

        Where x1 is greater than the median, compute SF(x1) - SF(x2),
        otherwise compute CDF(x2) - CDF(x1).

        This function is only useful if `dist.sf(x, ...)` has an implementation
        that is numerically more accurate than `1 - dist.cdf(x, ...)`.
        """
    cdf1 = self.cdf(x1, *args, loc=loc, scale=scale)
    result = np.where(cdf1 > 0.5, self.sf(x1, *args, loc=loc, scale=scale) - self.sf(x2, *args, loc=loc, scale=scale), self.cdf(x2, *args, loc=loc, scale=scale) - cdf1)
    if result.ndim == 0:
        result = result[()]
    return result