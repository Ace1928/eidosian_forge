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
class laplace_asymmetric_gen(rv_continuous):
    """An asymmetric Laplace continuous random variable.

    %(before_notes)s

    See Also
    --------
    laplace : Laplace distribution

    Notes
    -----
    The probability density function for `laplace_asymmetric` is

    .. math::

       f(x, \\kappa) &= \\frac{1}{\\kappa+\\kappa^{-1}}\\exp(-x\\kappa),\\quad x\\ge0\\\\
                    &= \\frac{1}{\\kappa+\\kappa^{-1}}\\exp(x/\\kappa),\\quad x<0\\\\

    for :math:`-\\infty < x < \\infty`, :math:`\\kappa > 0`.

    `laplace_asymmetric` takes ``kappa`` as a shape parameter for
    :math:`\\kappa`. For :math:`\\kappa = 1`, it is identical to a
    Laplace distribution.

    %(after_notes)s

    Note that the scale parameter of some references is the reciprocal of
    SciPy's ``scale``. For example, :math:`\\lambda = 1/2` in the
    parameterization of [1]_ is equivalent to ``scale = 2`` with
    `laplace_asymmetric`.

    References
    ----------
    .. [1] "Asymmetric Laplace distribution", Wikipedia
            https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution

    .. [2] Kozubowski TJ and Podgórski K. A Multivariate and
           Asymmetric Generalization of Laplace Distribution,
           Computational Statistics 15, 531--540 (2000).
           :doi:`10.1007/PL00022717`

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('kappa', False, (0, np.inf), (False, False))]

    def _pdf(self, x, kappa):
        return np.exp(self._logpdf(x, kappa))

    def _logpdf(self, x, kappa):
        kapinv = 1 / kappa
        lPx = x * np.where(x >= 0, -kappa, kapinv)
        lPx -= np.log(kappa + kapinv)
        return lPx

    def _cdf(self, x, kappa):
        kapinv = 1 / kappa
        kappkapinv = kappa + kapinv
        return np.where(x >= 0, 1 - np.exp(-x * kappa) * (kapinv / kappkapinv), np.exp(x * kapinv) * (kappa / kappkapinv))

    def _sf(self, x, kappa):
        kapinv = 1 / kappa
        kappkapinv = kappa + kapinv
        return np.where(x >= 0, np.exp(-x * kappa) * (kapinv / kappkapinv), 1 - np.exp(x * kapinv) * (kappa / kappkapinv))

    def _ppf(self, q, kappa):
        kapinv = 1 / kappa
        kappkapinv = kappa + kapinv
        return np.where(q >= kappa / kappkapinv, -np.log((1 - q) * kappkapinv * kappa) * kapinv, np.log(q * kappkapinv / kappa) * kappa)

    def _isf(self, q, kappa):
        kapinv = 1 / kappa
        kappkapinv = kappa + kapinv
        return np.where(q <= kapinv / kappkapinv, -np.log(q * kappkapinv * kappa) * kapinv, np.log((1 - q) * kappkapinv / kappa) * kappa)

    def _stats(self, kappa):
        kapinv = 1 / kappa
        mn = kapinv - kappa
        var = kapinv * kapinv + kappa * kappa
        g1 = 2.0 * (1 - np.power(kappa, 6)) / np.power(1 + np.power(kappa, 4), 1.5)
        g2 = 6.0 * (1 + np.power(kappa, 8)) / np.power(1 + np.power(kappa, 4), 2)
        return (mn, var, g1, g2)

    def _entropy(self, kappa):
        return 1 + np.log(kappa + 1 / kappa)