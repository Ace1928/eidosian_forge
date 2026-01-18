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
def _attach_argparser_methods(self):
    """
        Generates the argument-parsing functions dynamically and attaches
        them to the instance.

        Should be called from `_attach_methods`, typically in __init__ and
        during unpickling (__setstate__)
        """
    ns = {}
    exec(self._parse_arg_template, ns)
    for name in ['_parse_args', '_parse_args_stats', '_parse_args_rvs']:
        setattr(self, name, types.MethodType(ns[name], self))