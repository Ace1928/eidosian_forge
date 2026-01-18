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
@lambda func: np.vectorize(func.__get__(object), otypes=[np.float64])
@staticmethod
def _integrate_pdf(x0, x1, p, a, b):
    """
        Integrate the pdf of the genhyberbolic distribution from x0 to x1.
        This is a private function used by _cdf() and _sf() only; either x0
        will be -inf or x1 will be inf.
        """
    user_data = np.array([p, a, b], float).ctypes.data_as(ctypes.c_void_p)
    llc = LowLevelCallable.from_cython(_stats, '_genhyperbolic_pdf', user_data)
    d = np.sqrt((a + b) * (a - b))
    mean = b / d * sc.kv(p + 1, d) / sc.kv(p, d)
    epsrel = 1e-10
    epsabs = 0
    if x0 < mean < x1:
        intgrl = integrate.quad(llc, x0, mean, epsrel=epsrel, epsabs=epsabs)[0] + integrate.quad(llc, mean, x1, epsrel=epsrel, epsabs=epsabs)[0]
    else:
        intgrl = integrate.quad(llc, x0, x1, epsrel=epsrel, epsabs=epsabs)[0]
    if np.isnan(intgrl):
        msg = 'Infinite values encountered in scipy.special.kve. Values replaced by NaN to avoid incorrect results.'
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    return max(0.0, min(1.0, intgrl))