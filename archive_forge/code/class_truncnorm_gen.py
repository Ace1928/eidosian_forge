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
class truncnorm_gen(rv_continuous):
    """A truncated normal continuous random variable.

    %(before_notes)s

    Notes
    -----
    This distribution is the normal distribution centered on ``loc`` (default
    0), with standard deviation ``scale`` (default 1), and truncated at ``a``
    and ``b`` *standard deviations* from ``loc``. For arbitrary ``loc`` and
    ``scale``, ``a`` and ``b`` are *not* the abscissae at which the shifted
    and scaled distribution is truncated.

    .. note::
        If ``a_trunc`` and ``b_trunc`` are the abscissae at which we wish
        to truncate the distribution (as opposed to the number of standard
        deviations from ``loc``), then we can calculate the distribution
        parameters ``a`` and ``b`` as follows::

            a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale

        This is a common point of confusion. For additional clarification,
        please see the example below.

    %(example)s

    In the examples above, ``loc=0`` and ``scale=1``, so the plot is truncated
    at ``a`` on the left and ``b`` on the right. However, suppose we were to
    produce the same histogram with ``loc = 1`` and ``scale=0.5``.

    >>> loc, scale = 1, 0.5
    >>> rv = truncnorm(a, b, loc=loc, scale=scale)
    >>> x = np.linspace(truncnorm.ppf(0.01, a, b),
    ...                 truncnorm.ppf(0.99, a, b), 100)
    >>> r = rv.rvs(size=1000)

    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    >>> ax.set_xlim(a, b)
    >>> ax.legend(loc='best', frameon=False)
    >>> plt.show()

    Note that the distribution is no longer appears to be truncated at
    abscissae ``a`` and ``b``. That is because the *standard* normal
    distribution is first truncated at ``a`` and ``b``, *then* the resulting
    distribution is scaled by ``scale`` and shifted by ``loc``. If we instead
    want the shifted and scaled distribution to be truncated at ``a`` and
    ``b``, we need to transform these values before passing them as the
    distribution parameters.

    >>> a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale
    >>> rv = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale)
    >>> x = np.linspace(truncnorm.ppf(0.01, a, b),
    ...                 truncnorm.ppf(0.99, a, b), 100)
    >>> r = rv.rvs(size=10000)

    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    >>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    >>> ax.set_xlim(a-0.1, b+0.1)
    >>> ax.legend(loc='best', frameon=False)
    >>> plt.show()
    """

    def _argcheck(self, a, b):
        return a < b

    def _shape_info(self):
        ia = _ShapeInfo('a', False, (-np.inf, np.inf), (True, False))
        ib = _ShapeInfo('b', False, (-np.inf, np.inf), (False, True))
        return [ia, ib]

    def _fitstart(self, data):
        if isinstance(data, CensoredData):
            data = data._uncensor()
        return super()._fitstart(data, args=(np.min(data), np.max(data)))

    def _get_support(self, a, b):
        return (a, b)

    def _pdf(self, x, a, b):
        return np.exp(self._logpdf(x, a, b))

    def _logpdf(self, x, a, b):
        return _norm_logpdf(x) - _log_gauss_mass(a, b)

    def _cdf(self, x, a, b):
        return np.exp(self._logcdf(x, a, b))

    def _logcdf(self, x, a, b):
        x, a, b = np.broadcast_arrays(x, a, b)
        logcdf = np.asarray(_log_gauss_mass(a, x) - _log_gauss_mass(a, b))
        i = logcdf > -0.1
        if np.any(i):
            logcdf[i] = np.log1p(-np.exp(self._logsf(x[i], a[i], b[i])))
        return logcdf

    def _sf(self, x, a, b):
        return np.exp(self._logsf(x, a, b))

    def _logsf(self, x, a, b):
        x, a, b = np.broadcast_arrays(x, a, b)
        logsf = np.asarray(_log_gauss_mass(x, b) - _log_gauss_mass(a, b))
        i = logsf > -0.1
        if np.any(i):
            logsf[i] = np.log1p(-np.exp(self._logcdf(x[i], a[i], b[i])))
        return logsf

    def _entropy(self, a, b):
        A = _norm_cdf(a)
        B = _norm_cdf(b)
        Z = B - A
        C = np.log(np.sqrt(2 * np.pi * np.e) * Z)
        D = (a * _norm_pdf(a) - b * _norm_pdf(b)) / (2 * Z)
        h = C + D
        return h

    def _ppf(self, q, a, b):
        q, a, b = np.broadcast_arrays(q, a, b)
        case_left = a < 0
        case_right = ~case_left

        def ppf_left(q, a, b):
            log_Phi_x = _log_sum(_norm_logcdf(a), np.log(q) + _log_gauss_mass(a, b))
            return sc.ndtri_exp(log_Phi_x)

        def ppf_right(q, a, b):
            log_Phi_x = _log_sum(_norm_logcdf(-b), np.log1p(-q) + _log_gauss_mass(a, b))
            return -sc.ndtri_exp(log_Phi_x)
        out = np.empty_like(q)
        q_left = q[case_left]
        q_right = q[case_right]
        if q_left.size:
            out[case_left] = ppf_left(q_left, a[case_left], b[case_left])
        if q_right.size:
            out[case_right] = ppf_right(q_right, a[case_right], b[case_right])
        return out

    def _isf(self, q, a, b):
        q, a, b = np.broadcast_arrays(q, a, b)
        case_left = b < 0
        case_right = ~case_left

        def isf_left(q, a, b):
            log_Phi_x = _log_diff(_norm_logcdf(b), np.log(q) + _log_gauss_mass(a, b))
            return sc.ndtri_exp(np.real(log_Phi_x))

        def isf_right(q, a, b):
            log_Phi_x = _log_diff(_norm_logcdf(-a), np.log1p(-q) + _log_gauss_mass(a, b))
            return -sc.ndtri_exp(np.real(log_Phi_x))
        out = np.empty_like(q)
        q_left = q[case_left]
        q_right = q[case_right]
        if q_left.size:
            out[case_left] = isf_left(q_left, a[case_left], b[case_left])
        if q_right.size:
            out[case_right] = isf_right(q_right, a[case_right], b[case_right])
        return out

    def _munp(self, n, a, b):

        def n_th_moment(n, a, b):
            """
            Returns n-th moment. Defined only if n >= 0.
            Function cannot broadcast due to the loop over n
            """
            pA, pB = self._pdf(np.asarray([a, b]), a, b)
            probs = [pA, -pB]
            moments = [0, 1]
            for k in range(1, n + 1):
                vals = _lazywhere(probs, [probs, [a, b]], lambda x, y: x * y ** (k - 1), fillvalue=0)
                mk = np.sum(vals) + (k - 1) * moments[-2]
                moments.append(mk)
            return moments[-1]
        return _lazywhere((n >= 0) & (a == a) & (b == b), (n, a, b), np.vectorize(n_th_moment, otypes=[np.float64]), np.nan)

    def _stats(self, a, b, moments='mv'):
        pA, pB = self.pdf(np.array([a, b]), a, b)

        def _truncnorm_stats_scalar(a, b, pA, pB, moments):
            m1 = pA - pB
            mu = m1
            probs = [pA, -pB]
            vals = _lazywhere(probs, [probs, [a, b]], lambda x, y: x * y, fillvalue=0)
            m2 = 1 + np.sum(vals)
            vals = _lazywhere(probs, [probs, [a - mu, b - mu]], lambda x, y: x * y, fillvalue=0)
            mu2 = 1 + np.sum(vals)
            vals = _lazywhere(probs, [probs, [a, b]], lambda x, y: x * y ** 2, fillvalue=0)
            m3 = 2 * m1 + np.sum(vals)
            vals = _lazywhere(probs, [probs, [a, b]], lambda x, y: x * y ** 3, fillvalue=0)
            m4 = 3 * m2 + np.sum(vals)
            mu3 = m3 + m1 * (-3 * m2 + 2 * m1 ** 2)
            g1 = mu3 / np.power(mu2, 1.5)
            mu4 = m4 + m1 * (-4 * m3 + 3 * m1 * (2 * m2 - m1 ** 2))
            g2 = mu4 / mu2 ** 2 - 3
            return (mu, mu2, g1, g2)
        _truncnorm_stats = np.vectorize(_truncnorm_stats_scalar, excluded=('moments',))
        return _truncnorm_stats(a, b, pA, pB, moments)