import warnings
from collections.abc import Iterable
from functools import wraps, cached_property
import ctypes
import numpy as np
from numpy.polynomial import Polynomial
from scipy._lib.doccer import (extend_notes_in_docstring,
from scipy._lib._ccallback import LowLevelCallable
from scipy import optimize
from scipy import integrate
import scipy.special as sc
import scipy.special._ufuncs as scu
from scipy._lib._util import _lazyselect, _lazywhere
from . import _stats
from ._tukeylambda_stats import (tukeylambda_variance as _tlvar,
from ._distn_infrastructure import (
from ._ksstats import kolmogn, kolmognp, kolmogni
from ._constants import (_XMIN, _LOGXMIN, _EULER, _ZETA3, _SQRT_PI,
from ._censored_data import CensoredData
import scipy.stats._boost as _boost
from scipy.optimize import root_scalar
from scipy.stats._warnings_errors import FitError
import scipy.stats as stats
def _single_cdf(q, k, df):
    if df < 100000:
        cython_symbol = '_studentized_range_cdf'
        log_const = _stats._studentized_range_cdf_logconst(k, df)
        arg = [q, k, df, log_const]
        usr_data = np.array(arg, float).ctypes.data_as(ctypes.c_void_p)
        ranges = [(-np.inf, np.inf), (0, np.inf)]
    else:
        cython_symbol = '_studentized_range_cdf_asymptotic'
        arg = [q, k]
        usr_data = np.array(arg, float).ctypes.data_as(ctypes.c_void_p)
        ranges = [(-np.inf, np.inf)]
    llc = LowLevelCallable.from_cython(_stats, cython_symbol, usr_data)
    opts = dict(epsabs=1e-11, epsrel=1e-12)
    return integrate.nquad(llc, ranges=ranges, opts=opts)[0]