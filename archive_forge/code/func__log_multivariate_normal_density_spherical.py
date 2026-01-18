from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import linalg
def _log_multivariate_normal_density_spherical(x, means, covars):
    """Compute Gaussian log-density at x for a spherical model."""
    cv = covars.copy()
    if covars.ndim == 1:
        cv = cv[:, np.newaxis]
    if covars.shape[1] == 1:
        cv = np.tile(cv, (1, x.shape[-1]))
    return _log_multivariate_normal_density_diag(x, means, cv)