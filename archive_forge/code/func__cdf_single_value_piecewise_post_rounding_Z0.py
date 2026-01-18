import warnings
from functools import partial
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy.integrate._quadrature import _builtincoeffs
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
import scipy.special as sc
from scipy._lib._util import _lazywhere
from .._distn_infrastructure import rv_continuous, _ShapeInfo
from .._continuous_distns import uniform, expon, _norm_pdf, _norm_cdf
from .levyst import Nolan
from scipy._lib.doccer import inherit_docstring_from
def _cdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, quad_eps, x_tol_near_zeta):
    """Calculate cdf using Nolan's methods as detailed in [NO]."""
    _nolan = Nolan(alpha, beta, x0)
    zeta = _nolan.zeta
    xi = _nolan.xi
    c1 = _nolan.c1
    c3 = _nolan.c3
    g = _nolan.g
    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)
    if alpha == 1 and beta < 0 or x0 < zeta:
        return 1 - _cdf_single_value_piecewise_post_rounding_Z0(-x0, alpha, -beta, quad_eps, x_tol_near_zeta)
    elif x0 == zeta:
        return 0.5 - xi / np.pi
    if np.isclose(-xi, np.pi / 2, rtol=1e-14, atol=1e-14):
        return c1

    def integrand(theta):
        g_1 = g(theta)
        return np.exp(-g_1)
    with np.errstate(all='ignore'):
        left_support = -xi
        right_support = np.pi / 2
        if alpha > 1:
            if integrand(-xi) != 0.0:
                res = optimize.minimize(integrand, (-xi,), method='L-BFGS-B', bounds=[(-xi, np.pi / 2)])
                left_support = res.x[0]
        elif integrand(np.pi / 2) != 0.0:
            res = optimize.minimize(integrand, (np.pi / 2,), method='L-BFGS-B', bounds=[(-xi, np.pi / 2)])
            right_support = res.x[0]
        intg, *ret = integrate.quad(integrand, left_support, right_support, points=[left_support, right_support], limit=100, epsrel=quad_eps, epsabs=0, full_output=1)
    return c1 + c3 * intg