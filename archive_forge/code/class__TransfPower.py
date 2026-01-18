import warnings
import numpy as np
from scipy.special import expm1, gamma
class _TransfPower(Transforms):
    """generic multivariate Archimedean copula with additional power transforms

    Nelson p.144, equ. 4.5.2

    experimental, not yet tested and used
    """

    def __init__(self, transform):
        self.transform = transform

    def evaluate(self, t, alpha, beta, *tr_args):
        t = np.asarray(t)
        phi = np.power(self.transform.evaluate(np.power(t, alpha), *tr_args), beta)
        return phi

    def inverse(self, phi, alpha, beta, *tr_args):
        phi = np.asarray(phi)
        transf = self.transform
        phi_inv = np.power(transf.evaluate(np.power(phi, 1.0 / beta), *tr_args), 1.0 / alpha)
        return phi_inv