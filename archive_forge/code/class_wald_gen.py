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
class wald_gen(invgauss_gen):
    """A Wald continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `wald` is:

    .. math::

        f(x) = \\frac{1}{\\sqrt{2\\pi x^3}} \\exp(- \\frac{ (x-1)^2 }{ 2x })

    for :math:`x >= 0`.

    `wald` is a special case of `invgauss` with ``mu=1``.

    %(after_notes)s

    %(example)s
    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        return []

    def _rvs(self, size=None, random_state=None):
        return random_state.wald(1.0, 1.0, size=size)

    def _pdf(self, x):
        return invgauss._pdf(x, 1.0)

    def _cdf(self, x):
        return invgauss._cdf(x, 1.0)

    def _sf(self, x):
        return invgauss._sf(x, 1.0)

    def _ppf(self, x):
        return invgauss._ppf(x, 1.0)

    def _isf(self, x):
        return invgauss._isf(x, 1.0)

    def _logpdf(self, x):
        return invgauss._logpdf(x, 1.0)

    def _logcdf(self, x):
        return invgauss._logcdf(x, 1.0)

    def _logsf(self, x):
        return invgauss._logsf(x, 1.0)

    def _stats(self):
        return (1.0, 1.0, 3.0, 15.0)

    def _entropy(self):
        return invgauss._entropy(1.0)