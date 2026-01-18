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
class betaprime_gen(rv_continuous):
    """A beta prime continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `betaprime` is:

    .. math::

        f(x, a, b) = \\frac{x^{a-1} (1+x)^{-a-b}}{\\beta(a, b)}

    for :math:`x >= 0`, :math:`a > 0`, :math:`b > 0`, where
    :math:`\\beta(a, b)` is the beta function (see `scipy.special.beta`).

    `betaprime` takes ``a`` and ``b`` as shape parameters.

    The distribution is related to the `beta` distribution as follows:
    If :math:`X` follows a beta distribution with parameters :math:`a, b`,
    then :math:`Y = X/(1-X)` has a beta prime distribution with
    parameters :math:`a, b` ([1]_).

    The beta prime distribution is a reparametrized version of the
    F distribution.  The beta prime distribution with shape parameters
    ``a`` and ``b`` and ``scale = s`` is equivalent to the F distribution
    with parameters ``d1 = 2*a``, ``d2 = 2*b`` and ``scale = (a/b)*s``.
    For example,

    >>> from scipy.stats import betaprime, f
    >>> x = [1, 2, 5, 10]
    >>> a = 12
    >>> b = 5
    >>> betaprime.pdf(x, a, b, scale=2)
    array([0.00541179, 0.08331299, 0.14669185, 0.03150079])
    >>> f.pdf(x, 2*a, 2*b, scale=(a/b)*2)
    array([0.00541179, 0.08331299, 0.14669185, 0.03150079])

    %(after_notes)s

    References
    ----------
    .. [1] Beta prime distribution, Wikipedia,
           https://en.wikipedia.org/wiki/Beta_prime_distribution

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        ia = _ShapeInfo('a', False, (0, np.inf), (False, False))
        ib = _ShapeInfo('b', False, (0, np.inf), (False, False))
        return [ia, ib]

    def _rvs(self, a, b, size=None, random_state=None):
        u1 = gamma.rvs(a, size=size, random_state=random_state)
        u2 = gamma.rvs(b, size=size, random_state=random_state)
        return u1 / u2

    def _pdf(self, x, a, b):
        return np.exp(self._logpdf(x, a, b))

    def _logpdf(self, x, a, b):
        return sc.xlogy(a - 1.0, x) - sc.xlog1py(a + b, x) - sc.betaln(a, b)

    def _cdf(self, x, a, b):
        return _lazywhere(x > 1, [x, a, b], lambda x_, a_, b_: beta._sf(1 / (1 + x_), b_, a_), f2=lambda x_, a_, b_: beta._cdf(x_ / (1 + x_), a_, b_))

    def _sf(self, x, a, b):
        return _lazywhere(x > 1, [x, a, b], lambda x_, a_, b_: beta._cdf(1 / (1 + x_), b_, a_), f2=lambda x_, a_, b_: beta._sf(x_ / (1 + x_), a_, b_))

    def _ppf(self, p, a, b):
        p, a, b = np.broadcast_arrays(p, a, b)
        r = stats.beta._ppf(p, a, b)
        with np.errstate(divide='ignore'):
            out = r / (1 - r)
        i = r > 0.9999
        out[i] = 1 / stats.beta._isf(p[i], b[i], a[i]) - 1
        return out

    def _munp(self, n, a, b):
        return _lazywhere(b > n, (a, b), lambda a, b: np.prod([(a + i - 1) / (b - i) for i in range(1, n + 1)], axis=0), fillvalue=np.inf)