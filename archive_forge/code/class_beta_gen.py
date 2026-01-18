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
class beta_gen(rv_continuous):
    """A beta continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `beta` is:

    .. math::

        f(x, a, b) = \\frac{\\Gamma(a+b) x^{a-1} (1-x)^{b-1}}
                          {\\Gamma(a) \\Gamma(b)}

    for :math:`0 <= x <= 1`, :math:`a > 0`, :math:`b > 0`, where
    :math:`\\Gamma` is the gamma function (`scipy.special.gamma`).

    `beta` takes :math:`a` and :math:`b` as shape parameters.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        ia = _ShapeInfo('a', False, (0, np.inf), (False, False))
        ib = _ShapeInfo('b', False, (0, np.inf), (False, False))
        return [ia, ib]

    def _rvs(self, a, b, size=None, random_state=None):
        return random_state.beta(a, b, size)

    def _pdf(self, x, a, b):
        with np.errstate(over='ignore'):
            return _boost._beta_pdf(x, a, b)

    def _logpdf(self, x, a, b):
        lPx = sc.xlog1py(b - 1.0, -x) + sc.xlogy(a - 1.0, x)
        lPx -= sc.betaln(a, b)
        return lPx

    def _cdf(self, x, a, b):
        return _boost._beta_cdf(x, a, b)

    def _sf(self, x, a, b):
        return _boost._beta_sf(x, a, b)

    def _isf(self, x, a, b):
        with np.errstate(over='ignore'):
            return _boost._beta_isf(x, a, b)

    def _ppf(self, q, a, b):
        with np.errstate(over='ignore'):
            return _boost._beta_ppf(q, a, b)

    def _stats(self, a, b):
        return (_boost._beta_mean(a, b), _boost._beta_variance(a, b), _boost._beta_skewness(a, b), _boost._beta_kurtosis_excess(a, b))

    def _fitstart(self, data):
        if isinstance(data, CensoredData):
            data = data._uncensor()
        g1 = _skew(data)
        g2 = _kurtosis(data)

        def func(x):
            a, b = x
            sk = 2 * (b - a) * np.sqrt(a + b + 1) / (a + b + 2) / np.sqrt(a * b)
            ku = a ** 3 - a ** 2 * (2 * b - 1) + b ** 2 * (b + 1) - 2 * a * b * (b + 2)
            ku /= a * b * (a + b + 2) * (a + b + 3)
            ku *= 6
            return [sk - g1, ku - g2]
        a, b = optimize.fsolve(func, (1.0, 1.0))
        return super()._fitstart(data, args=(a, b))

    @_call_super_mom
    @extend_notes_in_docstring(rv_continuous, notes='        In the special case where `method="MLE"` and\n        both `floc` and `fscale` are given, a\n        `ValueError` is raised if any value `x` in `data` does not satisfy\n        `floc < x < floc + fscale`.\n\n')
    def fit(self, data, *args, **kwds):
        floc = kwds.get('floc', None)
        fscale = kwds.get('fscale', None)
        if floc is None or fscale is None:
            return super().fit(data, *args, **kwds)
        kwds.pop('floc', None)
        kwds.pop('fscale', None)
        f0 = _get_fixed_fit_value(kwds, ['f0', 'fa', 'fix_a'])
        f1 = _get_fixed_fit_value(kwds, ['f1', 'fb', 'fix_b'])
        _remove_optimizer_parameters(kwds)
        if f0 is not None and f1 is not None:
            raise ValueError('All parameters fixed. There is nothing to optimize.')
        if not np.isfinite(data).all():
            raise ValueError('The data contains non-finite values.')
        data = (np.ravel(data) - floc) / fscale
        if np.any(data <= 0) or np.any(data >= 1):
            raise FitDataError('beta', lower=floc, upper=floc + fscale)
        xbar = data.mean()
        if f0 is not None or f1 is not None:
            if f0 is not None:
                b = f0
                data = 1 - data
                xbar = 1 - xbar
            else:
                b = f1
            a = b * xbar / (1 - xbar)
            theta, info, ier, mesg = optimize.fsolve(_beta_mle_a, a, args=(b, len(data), np.log(data).sum()), full_output=True)
            if ier != 1:
                raise FitSolverError(mesg=mesg)
            a = theta[0]
            if f0 is not None:
                a, b = (b, a)
        else:
            s1 = np.log(data).sum()
            s2 = sc.log1p(-data).sum()
            fac = xbar * (1 - xbar) / data.var(ddof=0) - 1
            a = xbar * fac
            b = (1 - xbar) * fac
            theta, info, ier, mesg = optimize.fsolve(_beta_mle_ab, [a, b], args=(len(data), s1, s2), full_output=True)
            if ier != 1:
                raise FitSolverError(mesg=mesg)
            a, b = theta
        return (a, b, floc, fscale)

    def _entropy(self, a, b):

        def regular(a, b):
            return sc.betaln(a, b) - (a - 1) * sc.psi(a) - (b - 1) * sc.psi(b) + (a + b - 2) * sc.psi(a + b)

        def asymptotic_ab_large(a, b):
            sum_ab = a + b
            log_term = 0.5 * (np.log(2 * np.pi) + np.log(a) + np.log(b) - 3 * np.log(sum_ab) + 1)
            t1 = 110 / sum_ab + 20 * sum_ab ** (-2.0) + sum_ab ** (-3.0) - 2 * sum_ab ** (-4.0)
            t2 = -50 / a - 10 * a ** (-2.0) - a ** (-3.0) + a ** (-4.0)
            t3 = -50 / b - 10 * b ** (-2.0) - b ** (-3.0) + b ** (-4.0)
            return log_term + (t1 + t2 + t3) / 120

        def asymptotic_b_large(a, b):
            sum_ab = a + b
            t1 = sc.gammaln(a) - (a - 1) * sc.psi(a)
            t2 = -1 / (2 * b) + 1 / (12 * b) - b ** (-2.0) / 12 - b ** (-3.0) / 120 + b ** (-4.0) / 120 + b ** (-5.0) / 252 - b ** (-6.0) / 252 + 1 / sum_ab - 1 / (12 * sum_ab) + sum_ab ** (-2.0) / 6 + sum_ab ** (-3.0) / 120 - sum_ab ** (-4.0) / 60 - sum_ab ** (-5.0) / 252 + sum_ab ** (-6.0) / 126
            log_term = sum_ab * np.log1p(a / b) + np.log(b) - 2 * np.log(sum_ab)
            return t1 + t2 + log_term

        def threshold_large(v):
            if v == 1.0:
                return 1000
            j = np.log10(v)
            digits = int(j)
            d = int(v / 10 ** digits) + 2
            return d * 10 ** (7 + j)
        if a >= 4960000.0 and b >= 4960000.0:
            return asymptotic_ab_large(a, b)
        elif a <= 4900000.0 and b - a >= 1000000.0 and (b >= threshold_large(a)):
            return asymptotic_b_large(a, b)
        elif b <= 4900000.0 and a - b >= 1000000.0 and (a >= threshold_large(b)):
            return asymptotic_b_large(b, a)
        else:
            return regular(a, b)