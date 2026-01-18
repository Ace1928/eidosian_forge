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
def _rvs_Z1(alpha, beta, size=None, random_state=None):
    """Simulate random variables using Nolan's methods as detailed in [NO].
    """

    def alpha1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        return 2 / np.pi * ((np.pi / 2 + bTH) * tanTH - beta * np.log(np.pi / 2 * W * cosTH / (np.pi / 2 + bTH)))

    def beta0func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        return W / (cosTH / np.tan(aTH) + np.sin(TH)) * ((np.cos(aTH) + np.sin(aTH) * tanTH) / W) ** (1.0 / alpha)

    def otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        val0 = beta * np.tan(np.pi * alpha / 2)
        th0 = np.arctan(val0) / alpha
        val3 = W / (cosTH / np.tan(alpha * (th0 + TH)) + np.sin(TH))
        res3 = val3 * ((np.cos(aTH) + np.sin(aTH) * tanTH - val0 * (np.sin(aTH) - np.cos(aTH) * tanTH)) / W) ** (1.0 / alpha)
        return res3

    def alphanot1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        res = _lazywhere(beta == 0, (alpha, beta, TH, aTH, bTH, cosTH, tanTH, W), beta0func, f2=otherwise)
        return res
    alpha = np.broadcast_to(alpha, size)
    beta = np.broadcast_to(beta, size)
    TH = uniform.rvs(loc=-np.pi / 2.0, scale=np.pi, size=size, random_state=random_state)
    W = expon.rvs(size=size, random_state=random_state)
    aTH = alpha * TH
    bTH = beta * TH
    cosTH = np.cos(TH)
    tanTH = np.tan(TH)
    res = _lazywhere(alpha == 1, (alpha, beta, TH, aTH, bTH, cosTH, tanTH, W), alpha1func, f2=alphanot1func)
    return res