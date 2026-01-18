import warnings
from scipy import linalg, special
from scipy._lib._util import check_random_state
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, exp, pi,
import numpy as np
from . import _mvn
from ._stats import gaussian_kernel_estimate, gaussian_kernel_estimate_log
from scipy.special import logsumexp  # noqa: F401
def _compute_covariance(self):
    """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
    self.factor = self.covariance_factor()
    if not hasattr(self, '_data_cho_cov'):
        self._data_covariance = atleast_2d(cov(self.dataset, rowvar=1, bias=False, aweights=self.weights))
        self._data_cho_cov = linalg.cholesky(self._data_covariance, lower=True)
    self.covariance = self._data_covariance * self.factor ** 2
    self.cho_cov = (self._data_cho_cov * self.factor).astype(np.float64)
    self.log_det = 2 * np.log(np.diag(self.cho_cov * np.sqrt(2 * pi))).sum()