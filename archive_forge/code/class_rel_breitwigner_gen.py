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
class rel_breitwigner_gen(rv_continuous):
    """A relativistic Breit-Wigner random variable.

    %(before_notes)s

    See Also
    --------
    cauchy: Cauchy distribution, also known as the Breit-Wigner distribution.

    Notes
    -----

    The probability density function for `rel_breitwigner` is

    .. math::

        f(x, \\rho) = \\frac{k}{(x^2 - \\rho^2)^2 + \\rho^2}

    where

    .. math::
        k = \\frac{2\\sqrt{2}\\rho^2\\sqrt{\\rho^2 + 1}}
            {\\pi\\sqrt{\\rho^2 + \\rho\\sqrt{\\rho^2 + 1}}}

    The relativistic Breit-Wigner distribution is used in high energy physics
    to model resonances [1]_. It gives the uncertainty in the invariant mass,
    :math:`M` [2]_, of a resonance with characteristic mass :math:`M_0` and
    decay-width :math:`\\Gamma`, where :math:`M`, :math:`M_0` and :math:`\\Gamma`
    are expressed in natural units. In SciPy's parametrization, the shape
    parameter :math:`\\rho` is equal to :math:`M_0/\\Gamma` and takes values in
    :math:`(0, \\infty)`.

    Equivalently, the relativistic Breit-Wigner distribution is said to give
    the uncertainty in the center-of-mass energy :math:`E_{\\text{cm}}`. In
    natural units, the speed of light :math:`c` is equal to 1 and the invariant
    mass :math:`M` is equal to the rest energy :math:`Mc^2`. In the
    center-of-mass frame, the rest energy is equal to the total energy [3]_.

    %(after_notes)s

    :math:`\\rho = M/\\Gamma` and :math:`\\Gamma` is the scale parameter. For
    example, if one seeks to model the :math:`Z^0` boson with :math:`M_0
    \\approx 91.1876 \\text{ GeV}` and :math:`\\Gamma \\approx 2.4952\\text{ GeV}`
    [4]_ one can set ``rho=91.1876/2.4952`` and ``scale=2.4952``.

    To ensure a physically meaningful result when using the `fit` method, one
    should set ``floc=0`` to fix the location parameter to 0.

    References
    ----------
    .. [1] Relativistic Breit-Wigner distribution, Wikipedia,
           https://en.wikipedia.org/wiki/Relativistic_Breit-Wigner_distribution
    .. [2] Invariant mass, Wikipedia,
           https://en.wikipedia.org/wiki/Invariant_mass
    .. [3] Center-of-momentum frame, Wikipedia,
           https://en.wikipedia.org/wiki/Center-of-momentum_frame
    .. [4] M. Tanabashi et al. (Particle Data Group) Phys. Rev. D 98, 030001 -
           Published 17 August 2018

    %(example)s

    """

    def _argcheck(self, rho):
        return rho > 0

    def _shape_info(self):
        return [_ShapeInfo('rho', False, (0, np.inf), (False, False))]

    def _pdf(self, x, rho):
        C = np.sqrt(2 * (1 + 1 / rho ** 2) / (1 + np.sqrt(1 + 1 / rho ** 2))) * 2 / np.pi
        with np.errstate(over='ignore'):
            return C / (((x - rho) * (x + rho) / rho) ** 2 + 1)

    def _cdf(self, x, rho):
        C = np.sqrt(2 / (1 + np.sqrt(1 + 1 / rho ** 2))) / np.pi
        result = np.sqrt(-1 + 1j / rho) * np.arctan(x / np.sqrt(-rho * (rho + 1j)))
        result = C * 2 * np.imag(result)
        return np.clip(result, None, 1)

    def _munp(self, n, rho):
        if n == 1:
            C = np.sqrt(2 * (1 + 1 / rho ** 2) / (1 + np.sqrt(1 + 1 / rho ** 2))) / np.pi * rho
            return C * (np.pi / 2 + np.arctan(rho))
        if n == 2:
            C = np.sqrt((1 + 1 / rho ** 2) / (2 * (1 + np.sqrt(1 + 1 / rho ** 2)))) * rho
            result = (1 - rho * 1j) / np.sqrt(-1 - 1j / rho)
            return 2 * C * np.real(result)
        else:
            return np.inf

    def _stats(self, rho):
        return (None, None, np.nan, np.nan)

    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        data, _, floc, fscale = _check_fit_input_parameters(self, data, args, kwds)
        censored = isinstance(data, CensoredData)
        if censored:
            if data.num_censored() == 0:
                data = data._uncensored
                censored = False
        if floc is None or censored:
            return super().fit(data, *args, **kwds)
        if fscale is None:
            p25, p50, p75 = np.quantile(data - floc, [0.25, 0.5, 0.75])
            scale_0 = p75 - p25
            rho_0 = p50 / scale_0
            if not args:
                args = [rho_0]
            if 'scale' not in kwds:
                kwds['scale'] = scale_0
        else:
            M_0 = np.median(data - floc)
            rho_0 = M_0 / fscale
            if not args:
                args = [rho_0]
        return super().fit(data, *args, **kwds)