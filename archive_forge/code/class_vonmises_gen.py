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
class vonmises_gen(rv_continuous):
    """A Von Mises continuous random variable.

    %(before_notes)s

    See Also
    --------
    scipy.stats.vonmises_fisher : Von-Mises Fisher distribution on a
                                  hypersphere

    Notes
    -----
    The probability density function for `vonmises` and `vonmises_line` is:

    .. math::

        f(x, \\kappa) = \\frac{ \\exp(\\kappa \\cos(x)) }{ 2 \\pi I_0(\\kappa) }

    for :math:`-\\pi \\le x \\le \\pi`, :math:`\\kappa > 0`. :math:`I_0` is the
    modified Bessel function of order zero (`scipy.special.i0`).

    `vonmises` is a circular distribution which does not restrict the
    distribution to a fixed interval. Currently, there is no circular
    distribution framework in SciPy. The ``cdf`` is implemented such that
    ``cdf(x + 2*np.pi) == cdf(x) + 1``.

    `vonmises_line` is the same distribution, defined on :math:`[-\\pi, \\pi]`
    on the real line. This is a regular (i.e. non-circular) distribution.

    Note about distribution parameters: `vonmises` and `vonmises_line` take
    ``kappa`` as a shape parameter (concentration) and ``loc`` as the location
    (circular mean). A ``scale`` parameter is accepted but does not have any
    effect.

    Examples
    --------
    Import the necessary modules.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import vonmises

    Define distribution parameters.

    >>> loc = 0.5 * np.pi  # circular mean
    >>> kappa = 1  # concentration

    Compute the probability density at ``x=0`` via the ``pdf`` method.

    >>> vonmises.pdf(loc, kappa, 0)
    0.12570826359722018

    Verify that the percentile function ``ppf`` inverts the cumulative
    distribution function ``cdf`` up to floating point accuracy.

    >>> x = 1
    >>> cdf_value = vonmises.cdf(loc=loc, kappa=kappa, x=x)
    >>> ppf_value = vonmises.ppf(cdf_value, loc=loc, kappa=kappa)
    >>> x, cdf_value, ppf_value
    (1, 0.31489339900904967, 1.0000000000000004)

    Draw 1000 random variates by calling the ``rvs`` method.

    >>> number_of_samples = 1000
    >>> samples = vonmises(loc=loc, kappa=kappa).rvs(number_of_samples)

    Plot the von Mises density on a Cartesian and polar grid to emphasize
    that is is a circular distribution.

    >>> fig = plt.figure(figsize=(12, 6))
    >>> left = plt.subplot(121)
    >>> right = plt.subplot(122, projection='polar')
    >>> x = np.linspace(-np.pi, np.pi, 500)
    >>> vonmises_pdf = vonmises.pdf(loc, kappa, x)
    >>> ticks = [0, 0.15, 0.3]

    The left image contains the Cartesian plot.

    >>> left.plot(x, vonmises_pdf)
    >>> left.set_yticks(ticks)
    >>> number_of_bins = int(np.sqrt(number_of_samples))
    >>> left.hist(samples, density=True, bins=number_of_bins)
    >>> left.set_title("Cartesian plot")
    >>> left.set_xlim(-np.pi, np.pi)
    >>> left.grid(True)

    The right image contains the polar plot.

    >>> right.plot(x, vonmises_pdf, label="PDF")
    >>> right.set_yticks(ticks)
    >>> right.hist(samples, density=True, bins=number_of_bins,
    ...            label="Histogram")
    >>> right.set_title("Polar plot")
    >>> right.legend(bbox_to_anchor=(0.15, 1.06))

    """

    def _shape_info(self):
        return [_ShapeInfo('kappa', False, (0, np.inf), (False, False))]

    def _rvs(self, kappa, size=None, random_state=None):
        return random_state.vonmises(0.0, kappa, size=size)

    @inherit_docstring_from(rv_continuous)
    def rvs(self, *args, **kwds):
        rvs = super().rvs(*args, **kwds)
        return np.mod(rvs + np.pi, 2 * np.pi) - np.pi

    def _pdf(self, x, kappa):
        return np.exp(kappa * sc.cosm1(x)) / (2 * np.pi * sc.i0e(kappa))

    def _logpdf(self, x, kappa):
        return kappa * sc.cosm1(x) - np.log(2 * np.pi) - np.log(sc.i0e(kappa))

    def _cdf(self, x, kappa):
        return _stats.von_mises_cdf(kappa, x)

    def _stats_skip(self, kappa):
        return (0, None, 0, None)

    def _entropy(self, kappa):
        return -kappa * sc.i1e(kappa) / sc.i0e(kappa) + np.log(2 * np.pi * sc.i0e(kappa)) + kappa

    @extend_notes_in_docstring(rv_continuous, notes='        The default limits of integration are endpoints of the interval\n        of width ``2*pi`` centered at `loc` (e.g. ``[-pi, pi]`` when\n        ``loc=0``).\n\n')
    def expect(self, func=None, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds):
        _a, _b = (-np.pi, np.pi)
        if lb is None:
            lb = loc + _a
        if ub is None:
            ub = loc + _b
        return super().expect(func, args, loc, scale, lb, ub, conditional, **kwds)

    @_call_super_mom
    @extend_notes_in_docstring(rv_continuous, notes='        Fit data is assumed to represent angles and will be wrapped onto the\n        unit circle. `f0` and `fscale` are ignored; the returned shape is\n        always the maximum likelihood estimate and the scale is always\n        1. Initial guesses are ignored.\n\n')
    def fit(self, data, *args, **kwds):
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)
        data, fshape, floc, fscale = _check_fit_input_parameters(self, data, args, kwds)
        if self.a == -np.pi:
            return super().fit(data, *args, **kwds)
        data = np.mod(data, 2 * np.pi)

        def find_mu(data):
            return stats.circmean(data)

        def find_kappa(data, loc):
            r = np.sum(np.cos(loc - data)) / len(data)
            if r > 0:

                def solve_for_kappa(kappa):
                    return sc.i1e(kappa) / sc.i0e(kappa) - r
                root_res = root_scalar(solve_for_kappa, method='brentq', bracket=(np.finfo(float).tiny, 1e+16))
                return root_res.root
            else:
                return np.finfo(float).tiny
        loc = floc if floc is not None else find_mu(data)
        shape = fshape if fshape is not None else find_kappa(data, loc)
        loc = np.mod(loc + np.pi, 2 * np.pi) - np.pi
        return (shape, loc, 1)