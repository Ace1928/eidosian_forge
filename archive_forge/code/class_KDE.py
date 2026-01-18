from statsmodels.compat.python import lzip
import numpy as np
from statsmodels.tools.validation import array_like
from . import kernels
class KDE:
    """
    Kernel Density Estimator

    Parameters
    ----------
    x : array_like
        N-dimensional array from which the density is to be estimated
    kernel : Kernel Class
        Should be a class from *
    """

    def __init__(self, x, kernel=None):
        x = array_like(x, 'x', maxdim=2, contiguous=True)
        if x.ndim == 1:
            x = x[:, None]
        nobs, n_series = x.shape
        if kernel is None:
            kernel = kernels.Gaussian()
        if n_series > 1:
            if isinstance(kernel, kernels.CustomKernel):
                kernel = kernels.NdKernel(n_series, kernels=kernel)
        self.kernel = kernel
        self.n = n_series
        self.x = x

    def density(self, x):
        return self.kernel.density(self.x, x)

    def __call__(self, x, h='scott'):
        return np.array([self.density(xx) for xx in x])

    def evaluate(self, x, h='silverman'):
        density = self.kernel.density
        return np.array([density(xx) for xx in x])