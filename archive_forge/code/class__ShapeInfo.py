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
class _ShapeInfo:

    def __init__(self, name, integrality=False, domain=(-np.inf, np.inf), inclusive=(True, True)):
        self.name = name
        self.integrality = integrality
        domain = list(domain)
        if np.isfinite(domain[0]) and (not inclusive[0]):
            domain[0] = np.nextafter(domain[0], np.inf)
        if np.isfinite(domain[1]) and (not inclusive[1]):
            domain[1] = np.nextafter(domain[1], -np.inf)
        self.domain = domain