from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
class NdKernel:
    """Generic N-dimensial kernel

    Parameters
    ----------
    n : int
        The number of series for kernel estimates
    kernels : list
        kernels

    Can be constructed from either
    a) a list of n kernels which will be treated as
    indepent marginals on a gaussian copula (specified by H)
    or b) a single univariate kernel which will be applied radially to the
    mahalanobis distance defined by H.

    In the case of the Gaussian these are both equivalent, and the second constructiong
    is prefered.
    """

    def __init__(self, n, kernels=None, H=None):
        if kernels is None:
            kernels = Gaussian()
        self._kernels = kernels
        self.weights = None
        if H is None:
            H = np.matrix(np.identity(n))
        self._H = H
        self._Hrootinv = np.linalg.cholesky(H.I)

    def getH(self):
        """Getter for kernel bandwidth, H"""
        return self._H

    def setH(self, value):
        """Setter for kernel bandwidth, H"""
        self._H = value
    H = property(getH, setH, doc='Kernel bandwidth matrix')

    def density(self, xs, x):
        n = len(xs)
        if len(xs) > 0:
            if self.weights is not None:
                w = np.mean(self((xs - x) * self._Hrootinv).T * self.weights) / sum(self.weights)
            else:
                w = np.mean(self((xs - x) * self._Hrootinv))
            return w
        else:
            return np.nan

    def _kernweight(self, x):
        """returns the kernel weight for the independent multivariate kernel"""
        if isinstance(self._kernels, CustomKernel):
            x = np.asarray(x)
            d = (x * x).sum(-1)
            return self._kernels(np.asarray(d))

    def __call__(self, x):
        """
        This simply returns the value of the kernel function at x

        Does the same as weight if the function is normalised
        """
        return self._kernweight(x)