from scipy import stats, integrate, special
import numpy as np
class HPoly:
    """Orthonormal (for weight=1) Hermite Polynomial, uses finite bounds

    for current use with DensityOrthoPoly domain is defined as [-6,6]

    """

    def __init__(self, order):
        self.order = order
        from scipy.special import hermite
        self.poly = hermite(order)
        self.domain = (-6, +6)
        self.offsetfactor = 0.5

    def __call__(self, x):
        k = self.order
        lnfact = -(1.0 / 2) * (k * np.log(2.0) + special.gammaln(k + 1) + logpi2) - x * x / 2
        fact = np.exp(lnfact)
        return self.poly(x) * fact