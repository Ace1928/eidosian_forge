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
def _log_gauss_mass(a, b):
    """Log of Gaussian probability mass within an interval"""
    a, b = np.broadcast_arrays(a, b)
    case_left = b <= 0
    case_right = a > 0
    case_central = ~(case_left | case_right)

    def mass_case_left(a, b):
        return _log_diff(_norm_logcdf(b), _norm_logcdf(a))

    def mass_case_right(a, b):
        return mass_case_left(-b, -a)

    def mass_case_central(a, b):
        return sc.log1p(-_norm_cdf(a) - _norm_cdf(-b))
    out = np.full_like(a, fill_value=np.nan, dtype=np.complex128)
    if a[case_left].size:
        out[case_left] = mass_case_left(a[case_left], b[case_left])
    if a[case_right].size:
        out[case_right] = mass_case_right(a[case_right], b[case_right])
    if a[case_central].size:
        out[case_central] = mass_case_central(a[case_central], b[case_central])
    return np.real(out)