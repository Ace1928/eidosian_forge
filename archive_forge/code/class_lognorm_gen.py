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
class lognorm_gen(rv_continuous):
    """A lognormal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `lognorm` is:

    .. math::

        f(x, s) = \\frac{1}{s x \\sqrt{2\\pi}}
                  \\exp\\left(-\\frac{\\log^2(x)}{2s^2}\\right)

    for :math:`x > 0`, :math:`s > 0`.

    `lognorm` takes ``s`` as a shape parameter for :math:`s`.

    %(after_notes)s

    Suppose a normally distributed random variable ``X`` has  mean ``mu`` and
    standard deviation ``sigma``. Then ``Y = exp(X)`` is lognormally
    distributed with ``s = sigma`` and ``scale = exp(mu)``.

    %(example)s

    The logarithm of a log-normally distributed random variable is
    normally distributed:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats
    >>> fig, ax = plt.subplots(1, 1)
    >>> mu, sigma = 2, 0.5
    >>> X = stats.norm(loc=mu, scale=sigma)
    >>> Y = stats.lognorm(s=sigma, scale=np.exp(mu))
    >>> x = np.linspace(*X.interval(0.999))
    >>> y = Y.rvs(size=10000)
    >>> ax.plot(x, X.pdf(x), label='X (pdf)')
    >>> ax.hist(np.log(y), density=True, bins=x, label='log(Y) (histogram)')
    >>> ax.legend()
    >>> plt.show()

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        return [_ShapeInfo('s', False, (0, np.inf), (False, False))]

    def _rvs(self, s, size=None, random_state=None):
        return np.exp(s * random_state.standard_normal(size))

    def _pdf(self, x, s):
        return np.exp(self._logpdf(x, s))

    def _logpdf(self, x, s):
        return _lognorm_logpdf(x, s)

    def _cdf(self, x, s):
        return _norm_cdf(np.log(x) / s)

    def _logcdf(self, x, s):
        return _norm_logcdf(np.log(x) / s)

    def _ppf(self, q, s):
        return np.exp(s * _norm_ppf(q))

    def _sf(self, x, s):
        return _norm_sf(np.log(x) / s)

    def _logsf(self, x, s):
        return _norm_logsf(np.log(x) / s)

    def _isf(self, q, s):
        return np.exp(s * _norm_isf(q))

    def _stats(self, s):
        p = np.exp(s * s)
        mu = np.sqrt(p)
        mu2 = p * (p - 1)
        g1 = np.sqrt(p - 1) * (2 + p)
        g2 = np.polyval([1, 2, 3, 0, -6.0], p)
        return (mu, mu2, g1, g2)

    def _entropy(self, s):
        return 0.5 * (1 + np.log(2 * np.pi) + 2 * np.log(s))

    @_call_super_mom
    @extend_notes_in_docstring(rv_continuous, notes="        When `method='MLE'` and\n        the location parameter is fixed by using the `floc` argument,\n        this function uses explicit formulas for the maximum likelihood\n        estimation of the log-normal shape and scale parameters, so the\n        `optimizer`, `loc` and `scale` keyword arguments are ignored.\n        If the location is free, a likelihood maximum is found by\n        setting its partial derivative wrt to location to 0, and\n        solving by substituting the analytical expressions of shape\n        and scale (or provided parameters).\n        See, e.g., equation 3.1 in\n        A. Clifford Cohen & Betty Jones Whitten (1980)\n        Estimation in the Three-Parameter Lognormal Distribution,\n        Journal of the American Statistical Association, 75:370, 399-404\n        https://doi.org/10.2307/2287466\n        \n\n")
    def fit(self, data, *args, **kwds):
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)
        parameters = _check_fit_input_parameters(self, data, args, kwds)
        data, fshape, floc, fscale = parameters
        data_min = np.min(data)

        def get_shape_scale(loc):
            if fshape is None or fscale is None:
                lndata = np.log(data - loc)
            scale = fscale or np.exp(lndata.mean())
            shape = fshape or np.sqrt(np.mean((lndata - np.log(scale)) ** 2))
            return (shape, scale)

        def dL_dLoc(loc):
            shape, scale = get_shape_scale(loc)
            shifted = data - loc
            return np.sum((1 + np.log(shifted / scale) / shape ** 2) / shifted)

        def ll(loc):
            shape, scale = get_shape_scale(loc)
            return -self.nnlf((shape, loc, scale), data)
        if floc is None:
            spacing = np.spacing(data_min)
            rbrack = data_min - spacing
            dL_dLoc_rbrack = dL_dLoc(rbrack)
            ll_rbrack = ll(rbrack)
            delta = 2 * spacing
            while dL_dLoc_rbrack >= -1e-06:
                rbrack = data_min - delta
                dL_dLoc_rbrack = dL_dLoc(rbrack)
                delta *= 2
            if not np.isfinite(rbrack) or not np.isfinite(dL_dLoc_rbrack):
                return super().fit(data, *args, **kwds)
            lbrack = np.minimum(np.nextafter(rbrack, -np.inf), rbrack - 1)
            dL_dLoc_lbrack = dL_dLoc(lbrack)
            delta = 2 * (rbrack - lbrack)
            while np.isfinite(lbrack) and np.isfinite(dL_dLoc_lbrack) and (np.sign(dL_dLoc_lbrack) == np.sign(dL_dLoc_rbrack)):
                lbrack = rbrack - delta
                dL_dLoc_lbrack = dL_dLoc(lbrack)
                delta *= 2
            if not np.isfinite(lbrack) or not np.isfinite(dL_dLoc_lbrack):
                return super().fit(data, *args, **kwds)
            res = root_scalar(dL_dLoc, bracket=(lbrack, rbrack))
            if not res.converged:
                return super().fit(data, *args, **kwds)
            ll_root = ll(res.root)
            loc = res.root if ll_root > ll_rbrack else data_min - spacing
        else:
            if floc >= data_min:
                raise FitDataError('lognorm', lower=0.0, upper=np.inf)
            loc = floc
        shape, scale = get_shape_scale(loc)
        if not (self._argcheck(shape) and scale > 0):
            return super().fit(data, *args, **kwds)
        return (shape, loc, scale)