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
class rice_gen(rv_continuous):
    """A Rice continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `rice` is:

    .. math::

        f(x, b) = x \\exp(- \\frac{x^2 + b^2}{2}) I_0(x b)

    for :math:`x >= 0`, :math:`b > 0`. :math:`I_0` is the modified Bessel
    function of order zero (`scipy.special.i0`).

    `rice` takes ``b`` as a shape parameter for :math:`b`.

    %(after_notes)s

    The Rice distribution describes the length, :math:`r`, of a 2-D vector with
    components :math:`(U+u, V+v)`, where :math:`U, V` are constant, :math:`u,
    v` are independent Gaussian random variables with standard deviation
    :math:`s`.  Let :math:`R = \\sqrt{U^2 + V^2}`. Then the pdf of :math:`r` is
    ``rice.pdf(x, R/s, scale=s)``.

    %(example)s

    """

    def _argcheck(self, b):
        return b >= 0

    def _shape_info(self):
        return [_ShapeInfo('b', False, (0, np.inf), (True, False))]

    def _rvs(self, b, size=None, random_state=None):
        t = b / np.sqrt(2) + random_state.standard_normal(size=(2,) + size)
        return np.sqrt((t * t).sum(axis=0))

    def _cdf(self, x, b):
        return sc.chndtr(np.square(x), 2, np.square(b))

    def _ppf(self, q, b):
        return np.sqrt(sc.chndtrix(q, 2, np.square(b)))

    def _pdf(self, x, b):
        return x * np.exp(-(x - b) * (x - b) / 2.0) * sc.i0e(x * b)

    def _munp(self, n, b):
        nd2 = n / 2.0
        n1 = 1 + nd2
        b2 = b * b / 2.0
        return 2.0 ** nd2 * np.exp(-b2) * sc.gamma(n1) * sc.hyp1f1(n1, 1, b2)