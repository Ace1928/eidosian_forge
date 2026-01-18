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
class nct_gen(rv_continuous):
    """A non-central Student's t continuous random variable.

    %(before_notes)s

    Notes
    -----
    If :math:`Y` is a standard normal random variable and :math:`V` is
    an independent chi-square random variable (`chi2`) with :math:`k` degrees
    of freedom, then

    .. math::

        X = \\frac{Y + c}{\\sqrt{V/k}}

    has a non-central Student's t distribution on the real line.
    The degrees of freedom parameter :math:`k` (denoted ``df`` in the
    implementation) satisfies :math:`k > 0` and the noncentrality parameter
    :math:`c` (denoted ``nc`` in the implementation) is a real number.

    %(after_notes)s

    %(example)s

    """

    def _argcheck(self, df, nc):
        return (df > 0) & (nc == nc)

    def _shape_info(self):
        idf = _ShapeInfo('df', False, (0, np.inf), (False, False))
        inc = _ShapeInfo('nc', False, (-np.inf, np.inf), (False, False))
        return [idf, inc]

    def _rvs(self, df, nc, size=None, random_state=None):
        n = norm.rvs(loc=nc, size=size, random_state=random_state)
        c2 = chi2.rvs(df, size=size, random_state=random_state)
        return n * np.sqrt(df) / np.sqrt(c2)

    def _pdf(self, x, df, nc):
        n = df * 1.0
        nc = nc * 1.0
        x2 = x * x
        ncx2 = nc * nc * x2
        fac1 = n + x2
        trm1 = n / 2.0 * np.log(n) + sc.gammaln(n + 1) - (n * np.log(2) + nc * nc / 2 + n / 2 * np.log(fac1) + sc.gammaln(n / 2))
        Px = np.exp(trm1)
        valF = ncx2 / (2 * fac1)
        trm1 = np.sqrt(2) * nc * x * sc.hyp1f1(n / 2 + 1, 1.5, valF) / np.asarray(fac1 * sc.gamma((n + 1) / 2))
        trm2 = sc.hyp1f1((n + 1) / 2, 0.5, valF) / np.asarray(np.sqrt(fac1) * sc.gamma(n / 2 + 1))
        Px *= trm1 + trm2
        return np.clip(Px, 0, None)

    def _cdf(self, x, df, nc):
        with np.errstate(over='ignore'):
            return np.clip(_boost._nct_cdf(x, df, nc), 0, 1)

    def _ppf(self, q, df, nc):
        with np.errstate(over='ignore'):
            return _boost._nct_ppf(q, df, nc)

    def _sf(self, x, df, nc):
        with np.errstate(over='ignore'):
            return np.clip(_boost._nct_sf(x, df, nc), 0, 1)

    def _isf(self, x, df, nc):
        with np.errstate(over='ignore'):
            return _boost._nct_isf(x, df, nc)

    def _stats(self, df, nc, moments='mv'):
        mu = _boost._nct_mean(df, nc)
        mu2 = _boost._nct_variance(df, nc)
        g1 = _boost._nct_skewness(df, nc) if 's' in moments else None
        g2 = _boost._nct_kurtosis_excess(df, nc) if 'k' in moments else None
        return (mu, mu2, g1, g2)