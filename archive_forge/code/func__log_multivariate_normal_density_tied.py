from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import linalg
def _log_multivariate_normal_density_tied(x, means, covars):
    """Compute Gaussian log-density at X for a tied model"""
    cv = np.tile(covars, (means.shape[0], 1, 1))
    return _log_multivariate_normal_density_full(x, means, cv)