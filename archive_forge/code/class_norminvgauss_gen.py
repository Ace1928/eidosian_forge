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
class norminvgauss_gen(rv_continuous):
    """A Normal Inverse Gaussian continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `norminvgauss` is:

    .. math::

        f(x, a, b) = \\frac{a \\, K_1(a \\sqrt{1 + x^2})}{\\pi \\sqrt{1 + x^2}} \\,
                     \\exp(\\sqrt{a^2 - b^2} + b x)

    where :math:`x` is a real number, the parameter :math:`a` is the tail
    heaviness and :math:`b` is the asymmetry parameter satisfying
    :math:`a > 0` and :math:`|b| <= a`.
    :math:`K_1` is the modified Bessel function of second kind
    (`scipy.special.k1`).

    %(after_notes)s

    A normal inverse Gaussian random variable `Y` with parameters `a` and `b`
    can be expressed as a normal mean-variance mixture:
    `Y = b * V + sqrt(V) * X` where `X` is `norm(0,1)` and `V` is
    `invgauss(mu=1/sqrt(a**2 - b**2))`. This representation is used
    to generate random variates.

    Another common parametrization of the distribution (see Equation 2.1 in
    [2]_) is given by the following expression of the pdf:

    .. math::

        g(x, \\alpha, \\beta, \\delta, \\mu) =
        \\frac{\\alpha\\delta K_1\\left(\\alpha\\sqrt{\\delta^2 + (x - \\mu)^2}\\right)}
        {\\pi \\sqrt{\\delta^2 + (x - \\mu)^2}} \\,
        e^{\\delta \\sqrt{\\alpha^2 - \\beta^2} + \\beta (x - \\mu)}

    In SciPy, this corresponds to
    `a = alpha * delta, b = beta * delta, loc = mu, scale=delta`.

    References
    ----------
    .. [1] O. Barndorff-Nielsen, "Hyperbolic Distributions and Distributions on
           Hyperbolae", Scandinavian Journal of Statistics, Vol. 5(3),
           pp. 151-157, 1978.

    .. [2] O. Barndorff-Nielsen, "Normal Inverse Gaussian Distributions and
           Stochastic Volatility Modelling", Scandinavian Journal of
           Statistics, Vol. 24, pp. 1-13, 1997.

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _argcheck(self, a, b):
        return (a > 0) & (np.absolute(b) < a)

    def _shape_info(self):
        ia = _ShapeInfo('a', False, (0, np.inf), (False, False))
        ib = _ShapeInfo('b', False, (-np.inf, np.inf), (False, False))
        return [ia, ib]

    def _fitstart(self, data):
        return super()._fitstart(data, args=(1, 0.5))

    def _pdf(self, x, a, b):
        gamma = np.sqrt(a ** 2 - b ** 2)
        fac1 = a / np.pi
        sq = np.hypot(1, x)
        return fac1 * sc.k1e(a * sq) * np.exp(b * x - a * sq + gamma) / sq

    def _sf(self, x, a, b):
        if np.isscalar(x):
            return integrate.quad(self._pdf, x, np.inf, args=(a, b))[0]
        else:
            a = np.atleast_1d(a)
            b = np.atleast_1d(b)
            result = []
            for x0, a0, b0 in zip(x, a, b):
                result.append(integrate.quad(self._pdf, x0, np.inf, args=(a0, b0))[0])
            return np.array(result)

    def _isf(self, q, a, b):

        def _isf_scalar(q, a, b):

            def eq(x, a, b, q):
                return self._sf(x, a, b) - q
            xm = self.mean(a, b)
            em = eq(xm, a, b, q)
            if em == 0:
                return xm
            if em > 0:
                delta = 1
                left = xm
                right = xm + delta
                while eq(right, a, b, q) > 0:
                    delta = 2 * delta
                    right = xm + delta
            else:
                delta = 1
                right = xm
                left = xm - delta
                while eq(left, a, b, q) < 0:
                    delta = 2 * delta
                    left = xm - delta
            result = optimize.brentq(eq, left, right, args=(a, b, q), xtol=self.xtol)
            return result
        if np.isscalar(q):
            return _isf_scalar(q, a, b)
        else:
            result = []
            for q0, a0, b0 in zip(q, a, b):
                result.append(_isf_scalar(q0, a0, b0))
            return np.array(result)

    def _rvs(self, a, b, size=None, random_state=None):
        gamma = np.sqrt(a ** 2 - b ** 2)
        ig = invgauss.rvs(mu=1 / gamma, size=size, random_state=random_state)
        return b * ig + np.sqrt(ig) * norm.rvs(size=size, random_state=random_state)

    def _stats(self, a, b):
        gamma = np.sqrt(a ** 2 - b ** 2)
        mean = b / gamma
        variance = a ** 2 / gamma ** 3
        skewness = 3.0 * b / (a * np.sqrt(gamma))
        kurtosis = 3.0 * (1 + 4 * b ** 2 / a ** 2) / gamma
        return (mean, variance, skewness, kurtosis)