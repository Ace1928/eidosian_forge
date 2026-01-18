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
class exponweib_gen(rv_continuous):
    """An exponentiated Weibull continuous random variable.

    %(before_notes)s

    See Also
    --------
    weibull_min, numpy.random.Generator.weibull

    Notes
    -----
    The probability density function for `exponweib` is:

    .. math::

        f(x, a, c) = a c [1-\\exp(-x^c)]^{a-1} \\exp(-x^c) x^{c-1}

    and its cumulative distribution function is:

    .. math::

        F(x, a, c) = [1-\\exp(-x^c)]^a

    for :math:`x > 0`, :math:`a > 0`, :math:`c > 0`.

    `exponweib` takes :math:`a` and :math:`c` as shape parameters:

    * :math:`a` is the exponentiation parameter,
      with the special case :math:`a=1` corresponding to the
      (non-exponentiated) Weibull distribution `weibull_min`.
    * :math:`c` is the shape parameter of the non-exponentiated Weibull law.

    %(after_notes)s

    References
    ----------
    https://en.wikipedia.org/wiki/Exponentiated_Weibull_distribution

    %(example)s

    """

    def _shape_info(self):
        ia = _ShapeInfo('a', False, (0, np.inf), (False, False))
        ic = _ShapeInfo('c', False, (0, np.inf), (False, False))
        return [ia, ic]

    def _pdf(self, x, a, c):
        return np.exp(self._logpdf(x, a, c))

    def _logpdf(self, x, a, c):
        negxc = -x ** c
        exm1c = -sc.expm1(negxc)
        logp = np.log(a) + np.log(c) + sc.xlogy(a - 1.0, exm1c) + negxc + sc.xlogy(c - 1.0, x)
        return logp

    def _cdf(self, x, a, c):
        exm1c = -sc.expm1(-x ** c)
        return exm1c ** a

    def _ppf(self, q, a, c):
        return (-sc.log1p(-q ** (1.0 / a))) ** np.asarray(1.0 / c)

    def _sf(self, x, a, c):
        return -_pow1pm1(-np.exp(-x ** c), a)

    def _isf(self, p, a, c):
        return (-np.log(-_pow1pm1(-p, 1 / a))) ** (1 / c)