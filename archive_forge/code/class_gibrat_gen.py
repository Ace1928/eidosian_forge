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
class gibrat_gen(rv_continuous):
    """A Gibrat continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `gibrat` is:

    .. math::

        f(x) = \\frac{1}{x \\sqrt{2\\pi}} \\exp(-\\frac{1}{2} (\\log(x))^2)

    `gibrat` is a special case of `lognorm` with ``s=1``.

    %(after_notes)s

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        return []

    def _rvs(self, size=None, random_state=None):
        return np.exp(random_state.standard_normal(size))

    def _pdf(self, x):
        return np.exp(self._logpdf(x))

    def _logpdf(self, x):
        return _lognorm_logpdf(x, 1.0)

    def _cdf(self, x):
        return _norm_cdf(np.log(x))

    def _ppf(self, q):
        return np.exp(_norm_ppf(q))

    def _sf(self, x):
        return _norm_sf(np.log(x))

    def _isf(self, p):
        return np.exp(_norm_isf(p))

    def _stats(self):
        p = np.e
        mu = np.sqrt(p)
        mu2 = p * (p - 1)
        g1 = np.sqrt(p - 1) * (2 + p)
        g2 = np.polyval([1, 2, 3, 0, -6.0], p)
        return (mu, mu2, g1, g2)

    def _entropy(self):
        return 0.5 * np.log(2 * np.pi) + 0.5