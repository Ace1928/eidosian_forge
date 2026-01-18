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
class gumbel_l_gen(rv_continuous):
    """A left-skewed Gumbel continuous random variable.

    %(before_notes)s

    See Also
    --------
    gumbel_r, gompertz, genextreme

    Notes
    -----
    The probability density function for `gumbel_l` is:

    .. math::

        f(x) = \\exp(x - e^x)

    The Gumbel distribution is sometimes referred to as a type I Fisher-Tippett
    distribution.  It is also related to the extreme value distribution,
    log-Weibull and Gompertz distributions.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return []

    def _pdf(self, x):
        return np.exp(self._logpdf(x))

    def _logpdf(self, x):
        return x - np.exp(x)

    def _cdf(self, x):
        return -sc.expm1(-np.exp(x))

    def _ppf(self, q):
        return np.log(-sc.log1p(-q))

    def _logsf(self, x):
        return -np.exp(x)

    def _sf(self, x):
        return np.exp(-np.exp(x))

    def _isf(self, x):
        return np.log(-np.log(x))

    def _stats(self):
        return (-_EULER, np.pi * np.pi / 6.0, -12 * np.sqrt(6) / np.pi ** 3 * _ZETA3, 12.0 / 5)

    def _entropy(self):
        return _EULER + 1.0

    @_call_super_mom
    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        if kwds.get('floc') is not None:
            kwds['floc'] = -kwds['floc']
        loc_r, scale_r = gumbel_r.fit(-np.asarray(data), *args, **kwds)
        return (-loc_r, scale_r)