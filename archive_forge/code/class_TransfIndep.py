import warnings
import numpy as np
from scipy.special import expm1, gamma
class TransfIndep(Transforms):

    def evaluate(self, t, *args):
        t = np.asarray(t)
        return -np.log(t)

    def inverse(self, phi, *args):
        phi = np.asarray(phi)
        return np.exp(-phi)

    def deriv(self, t, *args):
        t = np.asarray(t)
        return -1.0 / t

    def deriv2(self, t, *args):
        t = np.asarray(t)
        return 1.0 / t ** 2

    def deriv2_inverse(self, phi, *args):
        return np.exp(-phi)

    def deriv3_inverse(self, phi, *args):
        return -np.exp(-phi)

    def deriv4_inverse(self, phi, *args):
        return np.exp(-phi)