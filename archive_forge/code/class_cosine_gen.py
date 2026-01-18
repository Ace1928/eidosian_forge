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
class cosine_gen(rv_continuous):
    """A cosine continuous random variable.

    %(before_notes)s

    Notes
    -----
    The cosine distribution is an approximation to the normal distribution.
    The probability density function for `cosine` is:

    .. math::

        f(x) = \\frac{1}{2\\pi} (1+\\cos(x))

    for :math:`-\\pi \\le x \\le \\pi`.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return []

    def _pdf(self, x):
        return 1.0 / 2 / np.pi * (1 + np.cos(x))

    def _logpdf(self, x):
        c = np.cos(x)
        return _lazywhere(c != -1, (c,), lambda c: np.log1p(c) - np.log(2 * np.pi), fillvalue=-np.inf)

    def _cdf(self, x):
        return scu._cosine_cdf(x)

    def _sf(self, x):
        return scu._cosine_cdf(-x)

    def _ppf(self, p):
        return scu._cosine_invcdf(p)

    def _isf(self, p):
        return -scu._cosine_invcdf(p)

    def _stats(self):
        v = np.pi * np.pi / 3.0 - 2.0
        k = -6.0 * (np.pi ** 4 - 90) / (5.0 * (np.pi * np.pi - 6) ** 2)
        return (0.0, v, 0.0, k)

    def _entropy(self):
        return np.log(4 * np.pi) - 1.0