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
def _sum_finite(x):
    """
    For a 1D array x, return a tuple containing the sum of the
    finite values of x and the number of nonfinite values.

    This is a utility function used when evaluating the negative
    loglikelihood for a distribution and an array of samples.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats._distn_infrastructure import _sum_finite
    >>> tot, nbad = _sum_finite(np.array([-2, -np.inf, 5, 1]))
    >>> tot
    4.0
    >>> nbad
    1
    """
    finite_x = np.isfinite(x)
    bad_count = finite_x.size - np.count_nonzero(finite_x)
    return (np.sum(x[finite_x]), bad_count)