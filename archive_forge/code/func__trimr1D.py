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
def _trimr1D(a, low_limit, up_limit, low_inclusive, up_inclusive):
    n = a.count()
    idx = a.argsort()
    if low_limit:
        if low_inclusive:
            lowidx = int(low_limit * n)
        else:
            lowidx = int(np.round(low_limit * n))
        a[idx[:lowidx]] = masked
    if up_limit is not None:
        if up_inclusive:
            upidx = n - int(n * up_limit)
        else:
            upidx = n - int(np.round(n * up_limit))
        a[idx[upidx:]] = masked
    return a