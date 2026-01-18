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
class crystalball_gen(rv_continuous):
    """
    Crystalball distribution

    %(before_notes)s

    Notes
    -----
    The probability density function for `crystalball` is:

    .. math::

        f(x, \\beta, m) =  \\begin{cases}
                            N \\exp(-x^2 / 2),  &\\text{for } x > -\\beta\\\\
                            N A (B - x)^{-m}  &\\text{for } x \\le -\\beta
                          \\end{cases}

    where :math:`A = (m / |\\beta|)^m  \\exp(-\\beta^2 / 2)`,
    :math:`B = m/|\\beta| - |\\beta|` and :math:`N` is a normalisation constant.

    `crystalball` takes :math:`\\beta > 0` and :math:`m > 1` as shape
    parameters.  :math:`\\beta` defines the point where the pdf changes
    from a power-law to a Gaussian distribution.  :math:`m` is the power
    of the power-law tail.

    References
    ----------
    .. [1] "Crystal Ball Function",
           https://en.wikipedia.org/wiki/Crystal_Ball_function

    %(after_notes)s

    .. versionadded:: 0.19.0

    %(example)s
    """

    def _argcheck(self, beta, m):
        """
        Shape parameter bounds are m > 1 and beta > 0.
        """
        return (m > 1) & (beta > 0)

    def _shape_info(self):
        ibeta = _ShapeInfo('beta', False, (0, np.inf), (False, False))
        im = _ShapeInfo('m', False, (1, np.inf), (False, False))
        return [ibeta, im]

    def _fitstart(self, data):
        return super()._fitstart(data, args=(1, 1.5))

    def _pdf(self, x, beta, m):
        """
        Return PDF of the crystalball function.

                                            --
                                           | exp(-x**2 / 2),  for x > -beta
        crystalball.pdf(x, beta, m) =  N * |
                                           | A * (B - x)**(-m), for x <= -beta
                                            --
        """
        N = 1.0 / (m / beta / (m - 1) * np.exp(-beta ** 2 / 2.0) + _norm_pdf_C * _norm_cdf(beta))

        def rhs(x, beta, m):
            return np.exp(-x ** 2 / 2)

        def lhs(x, beta, m):
            return (m / beta) ** m * np.exp(-beta ** 2 / 2.0) * (m / beta - beta - x) ** (-m)
        return N * _lazywhere(x > -beta, (x, beta, m), f=rhs, f2=lhs)

    def _logpdf(self, x, beta, m):
        """
        Return the log of the PDF of the crystalball function.
        """
        N = 1.0 / (m / beta / (m - 1) * np.exp(-beta ** 2 / 2.0) + _norm_pdf_C * _norm_cdf(beta))

        def rhs(x, beta, m):
            return -x ** 2 / 2

        def lhs(x, beta, m):
            return m * np.log(m / beta) - beta ** 2 / 2 - m * np.log(m / beta - beta - x)
        return np.log(N) + _lazywhere(x > -beta, (x, beta, m), f=rhs, f2=lhs)

    def _cdf(self, x, beta, m):
        """
        Return CDF of the crystalball function
        """
        N = 1.0 / (m / beta / (m - 1) * np.exp(-beta ** 2 / 2.0) + _norm_pdf_C * _norm_cdf(beta))

        def rhs(x, beta, m):
            return m / beta * np.exp(-beta ** 2 / 2.0) / (m - 1) + _norm_pdf_C * (_norm_cdf(x) - _norm_cdf(-beta))

        def lhs(x, beta, m):
            return (m / beta) ** m * np.exp(-beta ** 2 / 2.0) * (m / beta - beta - x) ** (-m + 1) / (m - 1)
        return N * _lazywhere(x > -beta, (x, beta, m), f=rhs, f2=lhs)

    def _ppf(self, p, beta, m):
        N = 1.0 / (m / beta / (m - 1) * np.exp(-beta ** 2 / 2.0) + _norm_pdf_C * _norm_cdf(beta))
        pbeta = N * (m / beta) * np.exp(-beta ** 2 / 2) / (m - 1)

        def ppf_less(p, beta, m):
            eb2 = np.exp(-beta ** 2 / 2)
            C = m / beta * eb2 / (m - 1)
            N = 1 / (C + _norm_pdf_C * _norm_cdf(beta))
            return m / beta - beta - ((m - 1) * (m / beta) ** (-m) / eb2 * p / N) ** (1 / (1 - m))

        def ppf_greater(p, beta, m):
            eb2 = np.exp(-beta ** 2 / 2)
            C = m / beta * eb2 / (m - 1)
            N = 1 / (C + _norm_pdf_C * _norm_cdf(beta))
            return _norm_ppf(_norm_cdf(-beta) + 1 / _norm_pdf_C * (p / N - C))
        return _lazywhere(p < pbeta, (p, beta, m), f=ppf_less, f2=ppf_greater)

    def _munp(self, n, beta, m):
        """
        Returns the n-th non-central moment of the crystalball function.
        """
        N = 1.0 / (m / beta / (m - 1) * np.exp(-beta ** 2 / 2.0) + _norm_pdf_C * _norm_cdf(beta))

        def n_th_moment(n, beta, m):
            """
            Returns n-th moment. Defined only if n+1 < m
            Function cannot broadcast due to the loop over n
            """
            A = (m / beta) ** m * np.exp(-beta ** 2 / 2.0)
            B = m / beta - beta
            rhs = 2 ** ((n - 1) / 2.0) * sc.gamma((n + 1) / 2) * (1.0 + (-1) ** n * sc.gammainc((n + 1) / 2, beta ** 2 / 2))
            lhs = np.zeros(rhs.shape)
            for k in range(n + 1):
                lhs += sc.binom(n, k) * B ** (n - k) * (-1) ** k / (m - k - 1) * (m / beta) ** (-m + k + 1)
            return A * lhs + rhs
        return N * _lazywhere(n + 1 < m, (n, beta, m), np.vectorize(n_th_moment, otypes=[np.float64]), np.inf)