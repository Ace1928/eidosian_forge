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
def _rank1d(data, use_missing=False):
    n = data.count()
    rk = np.empty(data.size, dtype=float)
    idx = data.argsort()
    rk[idx[:n]] = np.arange(1, n + 1)
    if use_missing:
        rk[idx[n:]] = (n + 1) / 2.0
    else:
        rk[idx[n:]] = 0
    repeats = find_repeats(data.copy())
    for r in repeats[0]:
        condition = (data == r).filled(False)
        rk[condition] = rk[condition].mean()
    return rk