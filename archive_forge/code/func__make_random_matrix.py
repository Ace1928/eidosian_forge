import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from .base import (
from .exceptions import DataDimensionalityWarning
from .utils import check_random_state
from .utils._param_validation import Interval, StrOptions, validate_params
from .utils.extmath import safe_sparse_dot
from .utils.random import sample_without_replacement
from .utils.validation import check_array, check_is_fitted
def _make_random_matrix(self, n_components, n_features):
    """Generate the random projection matrix

        Parameters
        ----------
        n_components : int
            Dimensionality of the target projection space.

        n_features : int
            Dimensionality of the original source space.

        Returns
        -------
        components : sparse matrix of shape (n_components, n_features)
            The generated random matrix in CSR format.

        """
    random_state = check_random_state(self.random_state)
    self.density_ = _check_density(self.density, n_features)
    return _sparse_random_matrix(n_components, n_features, density=self.density_, random_state=random_state)