from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
class Triweight(CustomKernel):

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 1.09375 * (1 - x * x) ** 3, h=h, domain=[-1.0, 1.0], norm=1.0)
        self._L2Norm = 350.0 / 429.0
        self._kernel_var = 1.0 / 9
        self._order = 2