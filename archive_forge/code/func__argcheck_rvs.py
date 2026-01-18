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
def _argcheck_rvs(self, *args, **kwargs):
    size = kwargs.get('size', None)
    all_bcast = np.broadcast_arrays(*args)

    def squeeze_left(a):
        while a.ndim > 0 and a.shape[0] == 1:
            a = a[0]
        return a
    all_bcast = [squeeze_left(a) for a in all_bcast]
    bcast_shape = all_bcast[0].shape
    bcast_ndim = all_bcast[0].ndim
    if size is None:
        size_ = bcast_shape
    else:
        size_ = tuple(np.atleast_1d(size))
    ndiff = bcast_ndim - len(size_)
    if ndiff < 0:
        bcast_shape = (1,) * -ndiff + bcast_shape
    elif ndiff > 0:
        size_ = (1,) * ndiff + size_
    ok = all([bcdim == 1 or bcdim == szdim for bcdim, szdim in zip(bcast_shape, size_)])
    if not ok:
        raise ValueError(f'size does not match the broadcast shape of the parameters. {size}, {size_}, {bcast_shape}')
    param_bcast = all_bcast[:-2]
    loc_bcast = all_bcast[-2]
    scale_bcast = all_bcast[-1]
    return (param_bcast, loc_bcast, scale_bcast, size_)