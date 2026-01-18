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
class johnsonsu_gen(rv_continuous):
    """A Johnson SU continuous random variable.

    %(before_notes)s

    See Also
    --------
    johnsonsb

    Notes
    -----
    The probability density function for `johnsonsu` is:

    .. math::

        f(x, a, b) = \\frac{b}{\\sqrt{x^2 + 1}}
                     \\phi(a + b \\log(x + \\sqrt{x^2 + 1}))

    where :math:`x`, :math:`a`, and :math:`b` are real scalars; :math:`b > 0`.
    :math:`\\phi` is the pdf of the normal distribution.

    `johnsonsu` takes :math:`a` and :math:`b` as shape parameters.

    The first four central moments are calculated according to the formulas
    in [1]_.

    %(after_notes)s

    References
    ----------
    .. [1] Taylor Enterprises. "Johnson Family of Distributions".
       https://variation.com/wp-content/distribution_analyzer_help/hs126.htm

    %(example)s

    """

    def _argcheck(self, a, b):
        return (b > 0) & (a == a)

    def _shape_info(self):
        ia = _ShapeInfo('a', False, (-np.inf, np.inf), (False, False))
        ib = _ShapeInfo('b', False, (0, np.inf), (False, False))
        return [ia, ib]

    def _pdf(self, x, a, b):
        x2 = x * x
        trm = _norm_pdf(a + b * np.arcsinh(x))
        return b * 1.0 / np.sqrt(x2 + 1.0) * trm

    def _cdf(self, x, a, b):
        return _norm_cdf(a + b * np.arcsinh(x))

    def _ppf(self, q, a, b):
        return np.sinh((_norm_ppf(q) - a) / b)

    def _sf(self, x, a, b):
        return _norm_sf(a + b * np.arcsinh(x))

    def _isf(self, x, a, b):
        return np.sinh((_norm_isf(x) - a) / b)

    def _stats(self, a, b, moments='mv'):
        mu, mu2, g1, g2 = (None, None, None, None)
        bn2 = b ** (-2.0)
        expbn2 = np.exp(bn2)
        a_b = a / b
        if 'm' in moments:
            mu = -expbn2 ** 0.5 * np.sinh(a_b)
        if 'v' in moments:
            mu2 = 0.5 * sc.expm1(bn2) * (expbn2 * np.cosh(2 * a_b) + 1)
        if 's' in moments:
            t1 = expbn2 ** 0.5 * sc.expm1(bn2) ** 0.5
            t2 = 3 * np.sinh(a_b)
            t3 = expbn2 * (expbn2 + 2) * np.sinh(3 * a_b)
            denom = np.sqrt(2) * (1 + expbn2 * np.cosh(2 * a_b)) ** (3 / 2)
            g1 = -t1 * (t2 + t3) / denom
        if 'k' in moments:
            t1 = 3 + 6 * expbn2
            t2 = 4 * expbn2 ** 2 * (expbn2 + 2) * np.cosh(2 * a_b)
            t3 = expbn2 ** 2 * np.cosh(4 * a_b)
            t4 = -3 + 3 * expbn2 ** 2 + 2 * expbn2 ** 3 + expbn2 ** 4
            denom = 2 * (1 + expbn2 * np.cosh(2 * a_b)) ** 2
            g2 = (t1 + t2 + t3 * t4) / denom - 3
        return (mu, mu2, g1, g2)