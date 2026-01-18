from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
class Epanechnikov(CustomKernel):

    def __init__(self, h=1.0):
        CustomKernel.__init__(self, shape=lambda x: 0.75 * (1 - x * x), h=h, domain=[-1.0, 1.0], norm=1.0)
        self._L2Norm = 0.6
        self._kernel_var = 0.2
        self._order = 2