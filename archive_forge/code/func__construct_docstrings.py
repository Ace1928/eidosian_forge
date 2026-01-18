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
def _construct_docstrings(self, name, longname):
    if name is None:
        name = 'Distribution'
    self.name = name
    if longname is None:
        if name[0] in ['aeiouAEIOU']:
            hstr = 'An '
        else:
            hstr = 'A '
        longname = hstr + name
    if sys.flags.optimize < 2:
        if self.__doc__ is None:
            self._construct_default_doc(longname=longname, docdict=docdict_discrete, discrete='discrete')
        else:
            dct = dict(distdiscrete)
            self._construct_doc(docdict_discrete, dct.get(self.name))
        self.__doc__ = self.__doc__.replace('\n    scale : array_like, optional\n        scale parameter (default=1)', '')