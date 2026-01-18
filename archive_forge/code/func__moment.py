import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import (
def _moment(a, moment, axis, *, mean=None):
    if np.abs(moment - np.round(moment)) > 0:
        raise ValueError('All moment parameters must be integers')
    if moment == 0 or moment == 1:
        shape = list(a.shape)
        del shape[axis]
        dtype = a.dtype.type if a.dtype.kind in 'fc' else np.float64
        if len(shape) == 0:
            return dtype(1.0 if moment == 0 else 0.0)
        else:
            return ma.ones(shape, dtype=dtype) if moment == 0 else ma.zeros(shape, dtype=dtype)
    else:
        n_list = [moment]
        current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n - 1) / 2
            else:
                current_n /= 2
            n_list.append(current_n)
        mean = a.mean(axis, keepdims=True) if mean is None else mean
        a_zero_mean = a - mean
        if n_list[-1] == 1:
            s = a_zero_mean.copy()
        else:
            s = a_zero_mean ** 2
        for n in n_list[-2::-1]:
            s = s ** 2
            if n % 2:
                s *= a_zero_mean
        return s.mean(axis)