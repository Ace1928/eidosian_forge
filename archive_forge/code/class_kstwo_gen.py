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
class kstwo_gen(rv_continuous):
    """Kolmogorov-Smirnov two-sided test statistic distribution.

    This is the distribution of the two-sided Kolmogorov-Smirnov (KS)
    statistic :math:`D_n` for a finite sample size ``n >= 1``
    (the shape parameter).

    %(before_notes)s

    See Also
    --------
    kstwobign, ksone, kstest

    Notes
    -----
    :math:`D_n` is given by

    .. math::

        D_n = \\text{sup}_x |F_n(x) - F(x)|

    where :math:`F` is a (continuous) CDF and :math:`F_n` is an empirical CDF.
    `kstwo` describes the distribution under the null hypothesis of the KS test
    that the empirical CDF corresponds to :math:`n` i.i.d. random variates
    with CDF :math:`F`.

    %(after_notes)s

    References
    ----------
    .. [1] Simard, R., L'Ecuyer, P. "Computing the Two-Sided
       Kolmogorov-Smirnov Distribution",  Journal of Statistical Software,
       Vol 39, 11, 1-18 (2011).

    %(example)s

    """

    def _argcheck(self, n):
        return (n >= 1) & (n == np.round(n))

    def _shape_info(self):
        return [_ShapeInfo('n', True, (1, np.inf), (True, False))]

    def _get_support(self, n):
        return (0.5 / (n if not isinstance(n, Iterable) else np.asanyarray(n)), 1.0)

    def _pdf(self, x, n):
        return kolmognp(n, x)

    def _cdf(self, x, n):
        return kolmogn(n, x)

    def _sf(self, x, n):
        return kolmogn(n, x, cdf=False)

    def _ppf(self, q, n):
        return kolmogni(n, q, cdf=True)

    def _isf(self, q, n):
        return kolmogni(n, q, cdf=False)