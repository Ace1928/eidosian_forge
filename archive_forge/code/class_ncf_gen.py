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
class ncf_gen(rv_continuous):
    """A non-central F distribution continuous random variable.

    %(before_notes)s

    See Also
    --------
    scipy.stats.f : Fisher distribution

    Notes
    -----
    The probability density function for `ncf` is:

    .. math::

        f(x, n_1, n_2, \\lambda) =
            \\exp\\left(\\frac{\\lambda}{2} +
                      \\lambda n_1 \\frac{x}{2(n_1 x + n_2)}
                \\right)
            n_1^{n_1/2} n_2^{n_2/2} x^{n_1/2 - 1} \\\\
            (n_2 + n_1 x)^{-(n_1 + n_2)/2}
            \\gamma(n_1/2) \\gamma(1 + n_2/2) \\\\
            \\frac{L^{\\frac{n_1}{2}-1}_{n_2/2}
                \\left(-\\lambda n_1 \\frac{x}{2(n_1 x + n_2)}\\right)}
            {B(n_1/2, n_2/2)
                \\gamma\\left(\\frac{n_1 + n_2}{2}\\right)}

    for :math:`n_1, n_2 > 0`, :math:`\\lambda \\ge 0`.  Here :math:`n_1` is the
    degrees of freedom in the numerator, :math:`n_2` the degrees of freedom in
    the denominator, :math:`\\lambda` the non-centrality parameter,
    :math:`\\gamma` is the logarithm of the Gamma function, :math:`L_n^k` is a
    generalized Laguerre polynomial and :math:`B` is the beta function.

    `ncf` takes ``df1``, ``df2`` and ``nc`` as shape parameters. If ``nc=0``,
    the distribution becomes equivalent to the Fisher distribution.

    %(after_notes)s

    %(example)s

    """

    def _argcheck(self, df1, df2, nc):
        return (df1 > 0) & (df2 > 0) & (nc >= 0)

    def _shape_info(self):
        idf1 = _ShapeInfo('df1', False, (0, np.inf), (False, False))
        idf2 = _ShapeInfo('df2', False, (0, np.inf), (False, False))
        inc = _ShapeInfo('nc', False, (0, np.inf), (True, False))
        return [idf1, idf2, inc]

    def _rvs(self, dfn, dfd, nc, size=None, random_state=None):
        return random_state.noncentral_f(dfn, dfd, nc, size)

    def _pdf(self, x, dfn, dfd, nc):
        return _boost._ncf_pdf(x, dfn, dfd, nc)

    def _cdf(self, x, dfn, dfd, nc):
        return _boost._ncf_cdf(x, dfn, dfd, nc)

    def _ppf(self, q, dfn, dfd, nc):
        with np.errstate(over='ignore'):
            return _boost._ncf_ppf(q, dfn, dfd, nc)

    def _sf(self, x, dfn, dfd, nc):
        return _boost._ncf_sf(x, dfn, dfd, nc)

    def _isf(self, x, dfn, dfd, nc):
        with np.errstate(over='ignore'):
            return _boost._ncf_isf(x, dfn, dfd, nc)

    def _munp(self, n, dfn, dfd, nc):
        val = (dfn * 1.0 / dfd) ** n
        term = sc.gammaln(n + 0.5 * dfn) + sc.gammaln(0.5 * dfd - n) - sc.gammaln(dfd * 0.5)
        val *= np.exp(-nc / 2.0 + term)
        val *= sc.hyp1f1(n + 0.5 * dfn, 0.5 * dfn, 0.5 * nc)
        return val

    def _stats(self, dfn, dfd, nc, moments='mv'):
        mu = _boost._ncf_mean(dfn, dfd, nc)
        mu2 = _boost._ncf_variance(dfn, dfd, nc)
        g1 = _boost._ncf_skewness(dfn, dfd, nc) if 's' in moments else None
        g2 = _boost._ncf_kurtosis_excess(dfn, dfd, nc) if 'k' in moments else None
        return (mu, mu2, g1, g2)