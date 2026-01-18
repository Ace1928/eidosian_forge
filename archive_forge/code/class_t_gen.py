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
class t_gen(rv_continuous):
    """A Student's t continuous random variable.

    For the noncentral t distribution, see `nct`.

    %(before_notes)s

    See Also
    --------
    nct

    Notes
    -----
    The probability density function for `t` is:

    .. math::

        f(x, \\nu) = \\frac{\\Gamma((\\nu+1)/2)}
                        {\\sqrt{\\pi \\nu} \\Gamma(\\nu/2)}
                    (1+x^2/\\nu)^{-(\\nu+1)/2}

    where :math:`x` is a real number and the degrees of freedom parameter
    :math:`\\nu` (denoted ``df`` in the implementation) satisfies
    :math:`\\nu > 0`. :math:`\\Gamma` is the gamma function
    (`scipy.special.gamma`).

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('df', False, (0, np.inf), (False, False))]

    def _rvs(self, df, size=None, random_state=None):
        return random_state.standard_t(df, size=size)

    def _pdf(self, x, df):
        return _lazywhere(df == np.inf, (x, df), f=lambda x, df: norm._pdf(x), f2=lambda x, df: np.exp(self._logpdf(x, df)))

    def _logpdf(self, x, df):

        def regular_formula(x, df):
            return sc.gammaln((df + 1) / 2) - sc.gammaln(df / 2) - 0.5 * np.log(df * np.pi) - (df + 1) / 2 * np.log1p(x * x / df)

        def asymptotic_formula(x, df):
            return -0.5 * (1 + np.log(2 * np.pi)) + df / 2 * np.log1p(1 / df) + 1 / 6 * (df + 1) ** (-1.0) - 1 / 45 * (df + 1) ** (-3.0) - 1 / 6 * df ** (-1.0) + 1 / 45 * df ** (-3.0) - (df + 1) / 2 * np.log1p(x * x / df)

        def norm_logpdf(x, df):
            return norm._logpdf(x)
        return _lazyselect((df == np.inf, (df >= 200) & np.isfinite(df), df < 200), (norm_logpdf, asymptotic_formula, regular_formula), (x, df))

    def _cdf(self, x, df):
        return sc.stdtr(df, x)

    def _sf(self, x, df):
        return sc.stdtr(df, -x)

    def _ppf(self, q, df):
        return sc.stdtrit(df, q)

    def _isf(self, q, df):
        return -sc.stdtrit(df, q)

    def _stats(self, df):
        infinite_df = np.isposinf(df)
        mu = np.where(df > 1, 0.0, np.inf)
        condlist = ((df > 1) & (df <= 2), (df > 2) & np.isfinite(df), infinite_df)
        choicelist = (lambda df: np.broadcast_to(np.inf, df.shape), lambda df: df / (df - 2.0), lambda df: np.broadcast_to(1, df.shape))
        mu2 = _lazyselect(condlist, choicelist, (df,), np.nan)
        g1 = np.where(df > 3, 0.0, np.nan)
        condlist = ((df > 2) & (df <= 4), (df > 4) & np.isfinite(df), infinite_df)
        choicelist = (lambda df: np.broadcast_to(np.inf, df.shape), lambda df: 6.0 / (df - 4.0), lambda df: np.broadcast_to(0, df.shape))
        g2 = _lazyselect(condlist, choicelist, (df,), np.nan)
        return (mu, mu2, g1, g2)

    def _entropy(self, df):
        if df == np.inf:
            return norm._entropy()

        def regular(df):
            half = df / 2
            half1 = (df + 1) / 2
            return half1 * (sc.digamma(half1) - sc.digamma(half)) + np.log(np.sqrt(df) * sc.beta(half, 0.5))

        def asymptotic(df):
            h = norm._entropy() + 1 / df + df ** (-2.0) / 4 - df ** (-3.0) / 6 - df ** (-4.0) / 8 + 3 / 10 * df ** (-5.0) + df ** (-6.0) / 4
            return h
        h = _lazywhere(df >= 100, (df,), f=asymptotic, f2=regular)
        return h