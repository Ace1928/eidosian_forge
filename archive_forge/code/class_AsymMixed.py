import numpy as np
from scipy import stats
from statsmodels.tools.numdiff import _approx_fprime_cs_scalar, approx_hess
class AsymMixed(PickandDependence):
    """asymmetric mixed model of Tawn 1988

    special case:  k=0, theta in [0,1] : symmetric mixed model of
        Tiago de Oliveira 1980

    restrictions:
     - theta > 0
     - theta + 3*k > 0
     - theta + k <= 1
     - theta + 2*k <= 1
    """
    k_args = 2

    def _check_args(self, theta, k):
        condth = theta >= 0
        cond1 = theta + 3 * k > 0 and theta + k <= 1 and (theta + 2 * k <= 1)
        return condth & cond1

    def evaluate(self, t, theta, k):
        transf = 1 - (theta + k) * t + theta * t * t + k * t ** 3
        return transf

    def deriv(self, t, theta, k):
        d_dt = -(theta + k) + 2 * theta * t + 3 * k * t ** 2
        return d_dt

    def deriv2(self, t, theta, k):
        d2_dt2 = 2 * theta + 6 * k * t
        return d2_dt2