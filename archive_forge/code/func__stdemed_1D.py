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
def _stdemed_1D(data):
    data = np.sort(data.compressed())
    n = len(data)
    z = 2.5758293035489004
    k = int(np.round((n + 1) / 2.0 - z * np.sqrt(n / 4.0), 0))
    return (data[n - k] - data[k - 1]) / (2.0 * z)