import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (
class GaussianMultivariateKernel(MultivariateKernel):
    """
    The Gaussian (squared exponential) multivariate kernel.
    """

    def call(self, x, loc):
        return np.exp(-(x - loc) ** 2 / (2 * self.bw2)).sum(1) / self.bwk