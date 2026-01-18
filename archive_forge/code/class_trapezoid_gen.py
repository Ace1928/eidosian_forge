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
class trapezoid_gen(rv_continuous):
    """A trapezoidal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The trapezoidal distribution can be represented with an up-sloping line
    from ``loc`` to ``(loc + c*scale)``, then constant to ``(loc + d*scale)``
    and then downsloping from ``(loc + d*scale)`` to ``(loc+scale)``.  This
    defines the trapezoid base from ``loc`` to ``(loc+scale)`` and the flat
    top from ``c`` to ``d`` proportional to the position along the base
    with ``0 <= c <= d <= 1``.  When ``c=d``, this is equivalent to `triang`
    with the same values for `loc`, `scale` and `c`.
    The method of [1]_ is used for computing moments.

    `trapezoid` takes :math:`c` and :math:`d` as shape parameters.

    %(after_notes)s

    The standard form is in the range [0, 1] with c the mode.
    The location parameter shifts the start to `loc`.
    The scale parameter changes the width from 1 to `scale`.

    %(example)s

    References
    ----------
    .. [1] Kacker, R.N. and Lawrence, J.F. (2007). Trapezoidal and triangular
       distributions for Type B evaluation of standard uncertainty.
       Metrologia 44, 117-127. :doi:`10.1088/0026-1394/44/2/003`


    """

    def _argcheck(self, c, d):
        return (c >= 0) & (c <= 1) & (d >= 0) & (d <= 1) & (d >= c)

    def _shape_info(self):
        ic = _ShapeInfo('c', False, (0, 1.0), (True, True))
        id = _ShapeInfo('d', False, (0, 1.0), (True, True))
        return [ic, id]

    def _pdf(self, x, c, d):
        u = 2 / (d - c + 1)
        return _lazyselect([x < c, (c <= x) & (x <= d), x > d], [lambda x, c, d, u: u * x / c, lambda x, c, d, u: u, lambda x, c, d, u: u * (1 - x) / (1 - d)], (x, c, d, u))

    def _cdf(self, x, c, d):
        return _lazyselect([x < c, (c <= x) & (x <= d), x > d], [lambda x, c, d: x ** 2 / c / (d - c + 1), lambda x, c, d: (c + 2 * (x - c)) / (d - c + 1), lambda x, c, d: 1 - (1 - x) ** 2 / (d - c + 1) / (1 - d)], (x, c, d))

    def _ppf(self, q, c, d):
        qc, qd = (self._cdf(c, c, d), self._cdf(d, c, d))
        condlist = [q < qc, q <= qd, q > qd]
        choicelist = [np.sqrt(q * c * (1 + d - c)), 0.5 * q * (1 + d - c) + 0.5 * c, 1 - np.sqrt((1 - q) * (d - c + 1) * (1 - d))]
        return np.select(condlist, choicelist)

    def _munp(self, n, c, d):
        ab_term = c ** (n + 1)
        dc_term = _lazyselect([d == 0.0, (0.0 < d) & (d < 1.0), d == 1.0], [lambda d: 1.0, lambda d: np.expm1((n + 2) * np.log(d)) / (d - 1.0), lambda d: n + 2], [d])
        val = 2.0 / (1.0 + d - c) * (dc_term - ab_term) / ((n + 1) * (n + 2))
        return val

    def _entropy(self, c, d):
        return 0.5 * (1.0 - d + c) / (1.0 + d - c) + np.log(0.5 * (1.0 + d - c))