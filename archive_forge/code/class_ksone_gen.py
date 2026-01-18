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
class ksone_gen(rv_continuous):
    """Kolmogorov-Smirnov one-sided test statistic distribution.

    This is the distribution of the one-sided Kolmogorov-Smirnov (KS)
    statistics :math:`D_n^+` and :math:`D_n^-`
    for a finite sample size ``n >= 1`` (the shape parameter).

    %(before_notes)s

    See Also
    --------
    kstwobign, kstwo, kstest

    Notes
    -----
    :math:`D_n^+` and :math:`D_n^-` are given by

    .. math::

        D_n^+ &= \\text{sup}_x (F_n(x) - F(x)),\\\\
        D_n^- &= \\text{sup}_x (F(x) - F_n(x)),\\\\

    where :math:`F` is a continuous CDF and :math:`F_n` is an empirical CDF.
    `ksone` describes the distribution under the null hypothesis of the KS test
    that the empirical CDF corresponds to :math:`n` i.i.d. random variates
    with CDF :math:`F`.

    %(after_notes)s

    References
    ----------
    .. [1] Birnbaum, Z. W. and Tingey, F.H. "One-sided confidence contours
       for probability distribution functions", The Annals of Mathematical
       Statistics, 22(4), pp 592-596 (1951).

    %(example)s

    """

    def _argcheck(self, n):
        return (n >= 1) & (n == np.round(n))

    def _shape_info(self):
        return [_ShapeInfo('n', True, (1, np.inf), (True, False))]

    def _pdf(self, x, n):
        return -scu._smirnovp(n, x)

    def _cdf(self, x, n):
        return scu._smirnovc(n, x)

    def _sf(self, x, n):
        return sc.smirnov(n, x)

    def _ppf(self, q, n):
        return scu._smirnovci(n, q)

    def _isf(self, q, n):
        return sc.smirnovi(n, q)