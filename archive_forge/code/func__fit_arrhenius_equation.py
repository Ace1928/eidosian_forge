from .._util import get_backend
from ..util.regression import least_squares
from ..util.pyutil import defaultnamedtuple
from ..units import default_constants, default_units, format_string, patched_numpy
def _fit_arrhenius_equation(T, k, kerr=None, linearized=False):
    """Curve fitting of the Arrhenius equation to data points

    Parameters
    ----------
    k : array_like
    T : float
    kerr : array_like (optional)
    linearized : bool

    """
    if len(k) != len(T):
        raise ValueError('k and T needs to be of equal length.')
    from math import exp
    import numpy as np
    p = np.polyfit(1 / T, np.log(k), 1)
    R = _get_R(constants=None, units=None)
    Ea = -R * p[0]
    A = exp(p[1])
    if linearized:
        return (A, Ea)
    from scipy.optimize import curve_fit
    if kerr is None:
        weights = None
    else:
        weights = 1 / kerr ** 2
    popt, pcov = curve_fit(arrhenius_equation, T, k, [A, Ea], weights)
    return (popt, pcov)