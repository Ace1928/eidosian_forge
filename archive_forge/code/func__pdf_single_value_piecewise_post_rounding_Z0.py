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
def _pdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, quad_eps, x_tol_near_zeta):
    """Calculate pdf using Nolan's methods as detailed in [NO]."""
    _nolan = Nolan(alpha, beta, x0)
    zeta = _nolan.zeta
    xi = _nolan.xi
    c2 = _nolan.c2
    g = _nolan.g
    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)
    if x0 == zeta:
        return sc.gamma(1 + 1 / alpha) * np.cos(xi) / np.pi / (1 + zeta ** 2) ** (1 / alpha / 2)
    elif x0 < zeta:
        return _pdf_single_value_piecewise_post_rounding_Z0(-x0, alpha, -beta, quad_eps, x_tol_near_zeta)
    if np.isclose(-xi, np.pi / 2, rtol=1e-14, atol=1e-14):
        return 0.0

    def integrand(theta):
        g_1 = g(theta)
        if not np.isfinite(g_1) or g_1 < 0:
            g_1 = 0
        return g_1 * np.exp(-g_1)
    with np.errstate(all='ignore'):
        peak = optimize.bisect(lambda t: g(t) - 1, -xi, np.pi / 2, xtol=quad_eps)
        tail_points = [optimize.bisect(lambda t: g(t) - exp_height, -xi, np.pi / 2) for exp_height in [100, 10, 5]]
        intg_points = [0, peak] + tail_points
        intg, *ret = integrate.quad(integrand, -xi, np.pi / 2, points=intg_points, limit=100, epsrel=quad_eps, epsabs=0, full_output=1)
    return c2 * intg