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
def _fit_loc_scale_support(self, data, *args):
    """Estimate loc and scale parameters from data accounting for support.

        Parameters
        ----------
        data : array_like
            Data to fit.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        Lhat : float
            Estimated location parameter for the data.
        Shat : float
            Estimated scale parameter for the data.

        """
    if isinstance(data, CensoredData):
        data = data._uncensor()
    else:
        data = np.asarray(data)
    loc_hat, scale_hat = self.fit_loc_scale(data, *args)
    self._argcheck(*args)
    _a, _b = self._get_support(*args)
    a, b = (_a, _b)
    support_width = b - a
    if support_width <= 0:
        return (loc_hat, scale_hat)
    a_hat = loc_hat + a * scale_hat
    b_hat = loc_hat + b * scale_hat
    data_a = np.min(data)
    data_b = np.max(data)
    if a_hat < data_a and data_b < b_hat:
        return (loc_hat, scale_hat)
    data_width = data_b - data_a
    rel_margin = 0.1
    margin = data_width * rel_margin
    if support_width < np.inf:
        loc_hat = data_a - a - margin
        scale_hat = (data_width + 2 * margin) / support_width
        return (loc_hat, scale_hat)
    if a > -np.inf:
        return (data_a - a - margin, 1)
    elif b < np.inf:
        return (data_b - b + margin, 1)
    else:
        raise RuntimeError