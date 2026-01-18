from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
class Cosine(CustomKernel):
    """
    Cosine Kernel

    K(u) = pi/4 cos(0.5 * pi * u) between -1.0 and 1.0
    """

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.7853981633974483 * np.cos(np.pi / 2.0 * x), h=h, domain=[-1.0, 1.0], norm=1.0)
        self._L2Norm = np.pi ** 2 / 16.0
        self._kernel_var = 0.1894305308612978
        self._order = 2