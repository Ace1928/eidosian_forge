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
class genpareto_gen(rv_continuous):
    """A generalized Pareto continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `genpareto` is:

    .. math::

        f(x, c) = (1 + c x)^{-1 - 1/c}

    defined for :math:`x \\ge 0` if :math:`c \\ge 0`, and for
    :math:`0 \\le x \\le -1/c` if :math:`c < 0`.

    `genpareto` takes ``c`` as a shape parameter for :math:`c`.

    For :math:`c=0`, `genpareto` reduces to the exponential
    distribution, `expon`:

    .. math::

        f(x, 0) = \\exp(-x)

    For :math:`c=-1`, `genpareto` is uniform on ``[0, 1]``:

    .. math::

        f(x, -1) = 1

    %(after_notes)s

    %(example)s

    """

    def _argcheck(self, c):
        return np.isfinite(c)

    def _shape_info(self):
        return [_ShapeInfo('c', False, (-np.inf, np.inf), (False, False))]

    def _get_support(self, c):
        c = np.asarray(c)
        b = _lazywhere(c < 0, (c,), lambda c: -1.0 / c, np.inf)
        a = np.where(c >= 0, self.a, self.a)
        return (a, b)

    def _pdf(self, x, c):
        return np.exp(self._logpdf(x, c))

    def _logpdf(self, x, c):
        return _lazywhere((x == x) & (c != 0), (x, c), lambda x, c: -sc.xlog1py(c + 1.0, c * x) / c, -x)

    def _cdf(self, x, c):
        return -sc.inv_boxcox1p(-x, -c)

    def _sf(self, x, c):
        return sc.inv_boxcox(-x, -c)

    def _logsf(self, x, c):
        return _lazywhere((x == x) & (c != 0), (x, c), lambda x, c: -sc.log1p(c * x) / c, -x)

    def _ppf(self, q, c):
        return -sc.boxcox1p(-q, -c)

    def _isf(self, q, c):
        return -sc.boxcox(q, -c)

    def _stats(self, c, moments='mv'):
        if 'm' not in moments:
            m = None
        else:
            m = _lazywhere(c < 1, (c,), lambda xi: 1 / (1 - xi), np.inf)
        if 'v' not in moments:
            v = None
        else:
            v = _lazywhere(c < 1 / 2, (c,), lambda xi: 1 / (1 - xi) ** 2 / (1 - 2 * xi), np.nan)
        if 's' not in moments:
            s = None
        else:
            s = _lazywhere(c < 1 / 3, (c,), lambda xi: 2 * (1 + xi) * np.sqrt(1 - 2 * xi) / (1 - 3 * xi), np.nan)
        if 'k' not in moments:
            k = None
        else:
            k = _lazywhere(c < 1 / 4, (c,), lambda xi: 3 * (1 - 2 * xi) * (2 * xi ** 2 + xi + 3) / (1 - 3 * xi) / (1 - 4 * xi) - 3, np.nan)
        return (m, v, s, k)

    def _munp(self, n, c):

        def __munp(n, c):
            val = 0.0
            k = np.arange(0, n + 1)
            for ki, cnk in zip(k, sc.comb(n, k)):
                val = val + cnk * (-1) ** ki / (1.0 - c * ki)
            return np.where(c * n < 1, val * (-1.0 / c) ** n, np.inf)
        return _lazywhere(c != 0, (c,), lambda c: __munp(n, c), sc.gamma(n + 1))

    def _entropy(self, c):
        return 1.0 + c