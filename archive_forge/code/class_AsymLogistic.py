import numpy as np
from scipy import stats
from statsmodels.tools.numdiff import _approx_fprime_cs_scalar, approx_hess
class AsymLogistic(PickandDependence):
    """asymmetric logistic model of Tawn 1988

    special case: a1=a2=1 : Gumbel

    restrictions:
     - theta in (0,1]
     - a1, a2 in [0,1]
    """
    k_args = 3

    def _check_args(self, a1, a2, theta):
        condth = theta > 0 and theta <= 1
        conda1 = a1 >= 0 and a1 <= 1
        conda2 = a2 >= 0 and a2 <= 1
        return condth and conda1 and conda2

    def evaluate(self, t, a1, a2, theta):
        transf = (1 - a2) * (1 - t)
        transf += (1 - a1) * t
        transf += ((a1 * t) ** (1.0 / theta) + (a2 * (1 - t)) ** (1.0 / theta)) ** theta
        return transf

    def deriv(self, t, a1, a2, theta):
        b = theta
        d1 = (a1 * (a1 * t) ** (1 / b - 1) - a2 * (a2 * (1 - t)) ** (1 / b - 1)) * ((a1 * t) ** (1 / b) + (a2 * (1 - t)) ** (1 / b)) ** (b - 1) - a1 + a2
        return d1

    def deriv2(self, t, a1, a2, theta):
        b = theta
        d2 = (1 - b) * (a1 * t) ** (1 / b) * (a2 * (1 - t)) ** (1 / b) * ((a1 * t) ** (1 / b) + (a2 * (1 - t)) ** (1 / b)) ** (b - 2) / (b * (1 - t) ** 2 * t ** 2)
        return d2