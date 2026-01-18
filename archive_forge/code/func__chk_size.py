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
def _chk_size(a, b):
    a = ma.asanyarray(a)
    b = ma.asanyarray(b)
    na, nb = (a.size, b.size)
    if na != nb:
        raise ValueError(f'The size of the input array should match! ({na} <> {nb})')
    return (a, b, na)