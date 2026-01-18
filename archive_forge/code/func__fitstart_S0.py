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
def _fitstart_S0(data):
    alpha, beta, delta1, gamma = _fitstart_S1(data)
    if alpha != 1:
        delta0 = delta1 + beta * gamma * np.tan(np.pi * alpha / 2.0)
    else:
        delta0 = delta1 + 2 * beta * gamma * np.log(gamma) / np.pi
    return (alpha, beta, delta0, gamma)