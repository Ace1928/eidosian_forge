from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
def _kernweight(self, x):
    """returns the kernel weight for the independent multivariate kernel"""
    if isinstance(self._kernels, CustomKernel):
        x = np.asarray(x)
        d = (x * x).sum(-1)
        return self._kernels(np.asarray(d))