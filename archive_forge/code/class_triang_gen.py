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
class triang_gen(rv_continuous):
    """A triangular continuous random variable.

    %(before_notes)s

    Notes
    -----
    The triangular distribution can be represented with an up-sloping line from
    ``loc`` to ``(loc + c*scale)`` and then downsloping for ``(loc + c*scale)``
    to ``(loc + scale)``.

    `triang` takes ``c`` as a shape parameter for :math:`0 \\le c \\le 1`.

    %(after_notes)s

    The standard form is in the range [0, 1] with c the mode.
    The location parameter shifts the start to `loc`.
    The scale parameter changes the width from 1 to `scale`.

    %(example)s

    """

    def _rvs(self, c, size=None, random_state=None):
        return random_state.triangular(0, c, 1, size)

    def _argcheck(self, c):
        return (c >= 0) & (c <= 1)

    def _shape_info(self):
        return [_ShapeInfo('c', False, (0, 1.0), (True, True))]

    def _pdf(self, x, c):
        r = _lazyselect([c == 0, x < c, (x >= c) & (c != 1), c == 1], [lambda x, c: 2 - 2 * x, lambda x, c: 2 * x / c, lambda x, c: 2 * (1 - x) / (1 - c), lambda x, c: 2 * x], (x, c))
        return r

    def _cdf(self, x, c):
        r = _lazyselect([c == 0, x < c, (x >= c) & (c != 1), c == 1], [lambda x, c: 2 * x - x * x, lambda x, c: x * x / c, lambda x, c: (x * x - 2 * x + c) / (c - 1), lambda x, c: x * x], (x, c))
        return r

    def _ppf(self, q, c):
        return np.where(q < c, np.sqrt(c * q), 1 - np.sqrt((1 - c) * (1 - q)))

    def _stats(self, c):
        return ((c + 1.0) / 3.0, (1.0 - c + c * c) / 18, np.sqrt(2) * (2 * c - 1) * (c + 1) * (c - 2) / (5 * np.power(1.0 - c + c * c, 1.5)), -3.0 / 5.0)

    def _entropy(self, c):
        return 0.5 - np.log(2)