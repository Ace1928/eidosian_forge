from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
class CovViaEigendecomposition(Covariance):

    def __init__(self, eigendecomposition):
        eigenvalues, eigenvectors = eigendecomposition
        eigenvalues = self._validate_vector(eigenvalues, 'eigenvalues')
        eigenvectors = self._validate_matrix(eigenvectors, 'eigenvectors')
        message = 'The shapes of `eigenvalues` and `eigenvectors` must be compatible.'
        try:
            eigenvalues = np.expand_dims(eigenvalues, -2)
            eigenvectors, eigenvalues = np.broadcast_arrays(eigenvectors, eigenvalues)
            eigenvalues = eigenvalues[..., 0, :]
        except ValueError:
            raise ValueError(message)
        i_zero = eigenvalues <= 0
        positive_eigenvalues = np.array(eigenvalues, dtype=np.float64)
        positive_eigenvalues[i_zero] = 1
        self._log_pdet = np.sum(np.log(positive_eigenvalues), axis=-1)
        psuedo_reciprocals = 1 / np.sqrt(positive_eigenvalues)
        psuedo_reciprocals[i_zero] = 0
        self._LP = eigenvectors * psuedo_reciprocals
        self._LA = eigenvectors * np.sqrt(eigenvalues)
        self._rank = positive_eigenvalues.shape[-1] - i_zero.sum(axis=-1)
        self._w = eigenvalues
        self._v = eigenvectors
        self._shape = eigenvectors.shape
        self._null_basis = eigenvectors * i_zero
        self._eps = _multivariate._eigvalsh_to_eps(eigenvalues) * 10 ** 3
        self._allow_singular = True

    def _whiten(self, x):
        return x @ self._LP

    def _colorize(self, x):
        return x @ self._LA.T

    @cached_property
    def _covariance(self):
        return self._v * self._w @ self._v.T

    def _support_mask(self, x):
        """
        Check whether x lies in the support of the distribution.
        """
        residual = np.linalg.norm(x @ self._null_basis, axis=-1)
        in_support = residual < self._eps
        return in_support