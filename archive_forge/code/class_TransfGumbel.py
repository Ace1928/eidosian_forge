import warnings
import numpy as np
from scipy.special import expm1, gamma
class TransfGumbel(Transforms):
    """
    requires theta >=1
    """

    def _checkargs(self, theta):
        return theta >= 1

    def evaluate(self, t, theta):
        return np.power(-np.log(t), theta)

    def inverse(self, phi, theta):
        return np.exp(-np.power(phi, 1.0 / theta))

    def deriv(self, t, theta):
        return -theta * (-np.log(t)) ** (theta - 1) / t

    def deriv2(self, t, theta):
        tmp1 = np.log(t)
        d2 = (theta * (-1) ** (1 + theta) * tmp1 ** (theta - 1) * (1 - theta) + theta * (-1) ** (1 + theta) * tmp1 ** theta) / (t ** 2 * tmp1)
        return d2

    def deriv2_inverse(self, phi, theta):
        th = theta
        d2 = (phi ** (2 / th) + (th - 1) * phi ** (1 / th)) / (phi ** 2 * th ** 2)
        d2 *= np.exp(-phi ** (1 / th))
        return d2

    def deriv3_inverse(self, phi, theta):
        p = phi
        b = theta
        d3 = (-p ** (3 / b) + (3 - 3 * b) * p ** (2 / b) + ((3 - 2 * b) * b - 1) * p ** (1 / b)) / (p * b) ** 3
        d3 *= np.exp(-p ** (1 / b))
        return d3

    def deriv4_inverse(self, phi, theta):
        p = phi
        b = theta
        d4 = ((6 * b ** 3 - 11 * b ** 2 + 6.0 * b - 1) * p ** (1 / b) + (11 * b ** 2 - 18 * b + 7) * p ** (2 / b) + 6 * (b - 1) * p ** (3 / b) + p ** (4 / b)) / (p * b) ** 4
        d4 *= np.exp(-p ** (1 / b))
        return d4

    def is_completly_monotonic(self, theta):
        return theta > 1