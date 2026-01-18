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
class halflogistic_gen(rv_continuous):
    """A half-logistic continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `halflogistic` is:

    .. math::

        f(x) = \\frac{ 2 e^{-x} }{ (1+e^{-x})^2 }
             = \\frac{1}{2} \\text{sech}(x/2)^2

    for :math:`x \\ge 0`.

    %(after_notes)s

    References
    ----------
    .. [1] Asgharzadeh et al (2011). "Comparisons of Methods of Estimation for the
           Half-Logistic Distribution". Selcuk J. Appl. Math. 93-108.

    %(example)s

    """

    def _shape_info(self):
        return []

    def _pdf(self, x):
        return np.exp(self._logpdf(x))

    def _logpdf(self, x):
        return np.log(2) - x - 2.0 * sc.log1p(np.exp(-x))

    def _cdf(self, x):
        return np.tanh(x / 2.0)

    def _ppf(self, q):
        return 2 * np.arctanh(q)

    def _sf(self, x):
        return 2 * sc.expit(-x)

    def _isf(self, q):
        return _lazywhere(q < 0.5, (q,), lambda q: -sc.logit(0.5 * q), f2=lambda q: 2 * np.arctanh(1 - q))

    def _munp(self, n):
        if n == 1:
            return 2 * np.log(2)
        if n == 2:
            return np.pi * np.pi / 3.0
        if n == 3:
            return 9 * _ZETA3
        if n == 4:
            return 7 * np.pi ** 4 / 15.0
        return 2 * (1 - pow(2.0, 1 - n)) * sc.gamma(n + 1) * sc.zeta(n, 1)

    def _entropy(self):
        return 2 - np.log(2)

    @_call_super_mom
    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)
        data, floc, fscale = _check_fit_input_parameters(self, data, args, kwds)

        def find_scale(data, loc):
            n_observations = data.shape[0]
            sorted_data = np.sort(data, axis=0)
            p = np.arange(1, n_observations + 1) / (n_observations + 1)
            q = 1 - p
            pp1 = 1 + p
            alpha = p - 0.5 * q * pp1 * np.log(pp1 / q)
            beta = 0.5 * q * pp1
            sorted_data = sorted_data - loc
            B = 2 * np.sum(alpha[1:] * sorted_data[1:])
            C = 2 * np.sum(beta[1:] * sorted_data[1:] ** 2)
            scale = (B + np.sqrt(B ** 2 + 8 * n_observations * C)) / (4 * n_observations)
            rtol = 1e-08
            relative_residual = 1
            shifted_mean = sorted_data.mean()
            while relative_residual > rtol:
                sum_term = sorted_data * sc.expit(-sorted_data / scale)
                scale_new = shifted_mean - 2 / n_observations * sum_term.sum()
                relative_residual = abs((scale - scale_new) / scale)
                scale = scale_new
            return scale
        data_min = np.min(data)
        if floc is not None:
            if data_min < floc:
                raise FitDataError('halflogistic', lower=floc, upper=np.inf)
            loc = floc
        else:
            loc = data_min
        scale = fscale if fscale is not None else find_scale(data, loc)
        return (loc, scale)