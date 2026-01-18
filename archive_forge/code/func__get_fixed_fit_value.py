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
def _get_fixed_fit_value(kwds, names):
    """
    Given names such as `['f0', 'fa', 'fix_a']`, check that there is
    at most one non-None value in `kwds` associaed with those names.
    Return that value, or None if none of the names occur in `kwds`.
    As a side effect, all occurrences of those names in `kwds` are
    removed.
    """
    vals = [(name, kwds.pop(name)) for name in names if name in kwds]
    if len(vals) > 1:
        repeated = [name for name, val in vals]
        raise ValueError('fit method got multiple keyword arguments to specify the same fixed parameter: ' + ', '.join(repeated))
    return vals[0][1] if vals else None