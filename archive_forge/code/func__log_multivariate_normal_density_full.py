from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import linalg
def _log_multivariate_normal_density_full(x, means, covars, min_covar=1e-07):
    """Log probability for full covariance matrices."""
    n_samples, n_dim = x.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim), lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, positive-definite")
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (x - mu).T, lower=True).T
        log_prob[:, c] = -0.5 * (np.sum(cv_sol ** 2, axis=1) + n_dim * np.log(2 * np.pi) + cv_log_det)
    return log_prob