import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def _freq_weights(weights):
    if weights is None:
        return weights
    int_weights = weights.astype(int)
    if (weights != int_weights).any():
        raise ValueError('frequency (integer count-type) weights required %s' % weights)
    return int_weights