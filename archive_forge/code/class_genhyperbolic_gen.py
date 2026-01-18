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
class genhyperbolic_gen(rv_continuous):
    """A generalized hyperbolic continuous random variable.

    %(before_notes)s

    See Also
    --------
    t, norminvgauss, geninvgauss, laplace, cauchy

    Notes
    -----
    The probability density function for `genhyperbolic` is:

    .. math::

        f(x, p, a, b) =
            \\frac{(a^2 - b^2)^{p/2}}
            {\\sqrt{2\\pi}a^{p-1/2}
            K_p\\Big(\\sqrt{a^2 - b^2}\\Big)}
            e^{bx} \\times \\frac{K_{p - 1/2}
            (a \\sqrt{1 + x^2})}
            {(\\sqrt{1 + x^2})^{1/2 - p}}

    for :math:`x, p \\in ( - \\infty; \\infty)`,
    :math:`|b| < a` if :math:`p \\ge 0`,
    :math:`|b| \\le a` if :math:`p < 0`.
    :math:`K_{p}(.)` denotes the modified Bessel function of the second
    kind and order :math:`p` (`scipy.special.kv`)

    `genhyperbolic` takes ``p`` as a tail parameter,
    ``a`` as a shape parameter,
    ``b`` as a skewness parameter.

    %(after_notes)s

    The original parameterization of the Generalized Hyperbolic Distribution
    is found in [1]_ as follows

    .. math::

        f(x, \\lambda, \\alpha, \\beta, \\delta, \\mu) =
           \\frac{(\\gamma/\\delta)^\\lambda}{\\sqrt{2\\pi}K_\\lambda(\\delta \\gamma)}
           e^{\\beta (x - \\mu)} \\times \\frac{K_{\\lambda - 1/2}
           (\\alpha \\sqrt{\\delta^2 + (x - \\mu)^2})}
           {(\\sqrt{\\delta^2 + (x - \\mu)^2} / \\alpha)^{1/2 - \\lambda}}

    for :math:`x \\in ( - \\infty; \\infty)`,
    :math:`\\gamma := \\sqrt{\\alpha^2 - \\beta^2}`,
    :math:`\\lambda, \\mu \\in ( - \\infty; \\infty)`,
    :math:`\\delta \\ge 0, |\\beta| < \\alpha` if :math:`\\lambda \\ge 0`,
    :math:`\\delta > 0, |\\beta| \\le \\alpha` if :math:`\\lambda < 0`.

    The location-scale-based parameterization implemented in
    SciPy is based on [2]_, where :math:`a = \\alpha\\delta`,
    :math:`b = \\beta\\delta`, :math:`p = \\lambda`,
    :math:`scale=\\delta` and :math:`loc=\\mu`

    Moments are implemented based on [3]_ and [4]_.

    For the distributions that are a special case such as Student's t,
    it is not recommended to rely on the implementation of genhyperbolic.
    To avoid potential numerical problems and for performance reasons,
    the methods of the specific distributions should be used.

    References
    ----------
    .. [1] O. Barndorff-Nielsen, "Hyperbolic Distributions and Distributions
       on Hyperbolae", Scandinavian Journal of Statistics, Vol. 5(3),
       pp. 151-157, 1978. https://www.jstor.org/stable/4615705

    .. [2] Eberlein E., Prause K. (2002) The Generalized Hyperbolic Model:
        Financial Derivatives and Risk Measures. In: Geman H., Madan D.,
        Pliska S.R., Vorst T. (eds) Mathematical Finance - Bachelier
        Congress 2000. Springer Finance. Springer, Berlin, Heidelberg.
        :doi:`10.1007/978-3-662-12429-1_12`

    .. [3] Scott, David J, Würtz, Diethelm, Dong, Christine and Tran,
       Thanh Tam, (2009), Moments of the generalized hyperbolic
       distribution, MPRA Paper, University Library of Munich, Germany,
       https://EconPapers.repec.org/RePEc:pra:mprapa:19081.

    .. [4] E. Eberlein and E. A. von Hammerstein. Generalized hyperbolic
       and inverse Gaussian distributions: Limiting cases and approximation
       of processes. FDM Preprint 80, April 2003. University of Freiburg.
       https://freidok.uni-freiburg.de/fedora/objects/freidok:7974/datastreams/FILE1/content

    %(example)s

    """

    def _argcheck(self, p, a, b):
        return np.logical_and(np.abs(b) < a, p >= 0) | np.logical_and(np.abs(b) <= a, p < 0)

    def _shape_info(self):
        ip = _ShapeInfo('p', False, (-np.inf, np.inf), (False, False))
        ia = _ShapeInfo('a', False, (0, np.inf), (True, False))
        ib = _ShapeInfo('b', False, (-np.inf, np.inf), (False, False))
        return [ip, ia, ib]

    def _fitstart(self, data):
        return super()._fitstart(data, args=(1, 1, 0.5))

    def _logpdf(self, x, p, a, b):

        @np.vectorize
        def _logpdf_single(x, p, a, b):
            return _stats.genhyperbolic_logpdf(x, p, a, b)
        return _logpdf_single(x, p, a, b)

    def _pdf(self, x, p, a, b):

        @np.vectorize
        def _pdf_single(x, p, a, b):
            return _stats.genhyperbolic_pdf(x, p, a, b)
        return _pdf_single(x, p, a, b)

    @lambda func: np.vectorize(func.__get__(object), otypes=[np.float64])
    @staticmethod
    def _integrate_pdf(x0, x1, p, a, b):
        """
        Integrate the pdf of the genhyberbolic distribution from x0 to x1.
        This is a private function used by _cdf() and _sf() only; either x0
        will be -inf or x1 will be inf.
        """
        user_data = np.array([p, a, b], float).ctypes.data_as(ctypes.c_void_p)
        llc = LowLevelCallable.from_cython(_stats, '_genhyperbolic_pdf', user_data)
        d = np.sqrt((a + b) * (a - b))
        mean = b / d * sc.kv(p + 1, d) / sc.kv(p, d)
        epsrel = 1e-10
        epsabs = 0
        if x0 < mean < x1:
            intgrl = integrate.quad(llc, x0, mean, epsrel=epsrel, epsabs=epsabs)[0] + integrate.quad(llc, mean, x1, epsrel=epsrel, epsabs=epsabs)[0]
        else:
            intgrl = integrate.quad(llc, x0, x1, epsrel=epsrel, epsabs=epsabs)[0]
        if np.isnan(intgrl):
            msg = 'Infinite values encountered in scipy.special.kve. Values replaced by NaN to avoid incorrect results.'
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
        return max(0.0, min(1.0, intgrl))

    def _cdf(self, x, p, a, b):
        return self._integrate_pdf(-np.inf, x, p, a, b)

    def _sf(self, x, p, a, b):
        return self._integrate_pdf(x, np.inf, p, a, b)

    def _rvs(self, p, a, b, size=None, random_state=None):
        t1 = np.float_power(a, 2) - np.float_power(b, 2)
        t2 = np.float_power(t1, 0.5)
        t3 = np.float_power(t1, -0.5)
        gig = geninvgauss.rvs(p=p, b=t2, scale=t3, size=size, random_state=random_state)
        normst = norm.rvs(size=size, random_state=random_state)
        return b * gig + np.sqrt(gig) * normst

    def _stats(self, p, a, b):
        p, a, b = np.broadcast_arrays(p, a, b)
        t1 = np.float_power(a, 2) - np.float_power(b, 2)
        t1 = np.float_power(t1, 0.5)
        t2 = np.float_power(1, 2) * np.float_power(t1, -1)
        integers = np.linspace(0, 4, 5)
        integers = integers.reshape(integers.shape + (1,) * p.ndim)
        b0, b1, b2, b3, b4 = sc.kv(p + integers, t1)
        r1, r2, r3, r4 = (b / b0 for b in (b1, b2, b3, b4))
        m = b * t2 * r1
        v = t2 * r1 + np.float_power(b, 2) * np.float_power(t2, 2) * (r2 - np.float_power(r1, 2))
        m3e = np.float_power(b, 3) * np.float_power(t2, 3) * (r3 - 3 * b2 * b1 * np.float_power(b0, -2) + 2 * np.float_power(r1, 3)) + 3 * b * np.float_power(t2, 2) * (r2 - np.float_power(r1, 2))
        s = m3e * np.float_power(v, -3 / 2)
        m4e = np.float_power(b, 4) * np.float_power(t2, 4) * (r4 - 4 * b3 * b1 * np.float_power(b0, -2) + 6 * b2 * np.float_power(b1, 2) * np.float_power(b0, -3) - 3 * np.float_power(r1, 4)) + np.float_power(b, 2) * np.float_power(t2, 3) * (6 * r3 - 12 * b2 * b1 * np.float_power(b0, -2) + 6 * np.float_power(r1, 3)) + 3 * np.float_power(t2, 2) * r2
        k = m4e * np.float_power(v, -2) - 3
        return (m, v, s, k)