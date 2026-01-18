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
class tukeylambda_gen(rv_continuous):
    """A Tukey-Lamdba continuous random variable.

    %(before_notes)s

    Notes
    -----
    A flexible distribution, able to represent and interpolate between the
    following distributions:

    - Cauchy                (:math:`lambda = -1`)
    - logistic              (:math:`lambda = 0`)
    - approx Normal         (:math:`lambda = 0.14`)
    - uniform from -1 to 1  (:math:`lambda = 1`)

    `tukeylambda` takes a real number :math:`lambda` (denoted ``lam``
    in the implementation) as a shape parameter.

    %(after_notes)s

    %(example)s

    """

    def _argcheck(self, lam):
        return np.isfinite(lam)

    def _shape_info(self):
        return [_ShapeInfo('lam', False, (-np.inf, np.inf), (False, False))]

    def _pdf(self, x, lam):
        Fx = np.asarray(sc.tklmbda(x, lam))
        Px = Fx ** (lam - 1.0) + np.asarray(1 - Fx) ** (lam - 1.0)
        Px = 1.0 / np.asarray(Px)
        return np.where((lam <= 0) | (abs(x) < 1.0 / np.asarray(lam)), Px, 0.0)

    def _cdf(self, x, lam):
        return sc.tklmbda(x, lam)

    def _ppf(self, q, lam):
        return sc.boxcox(q, lam) - sc.boxcox1p(-q, lam)

    def _stats(self, lam):
        return (0, _tlvar(lam), 0, _tlkurt(lam))

    def _entropy(self, lam):

        def integ(p):
            return np.log(pow(p, lam - 1) + pow(1 - p, lam - 1))
        return integrate.quad(integ, 0, 1)[0]