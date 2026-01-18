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
def _gaussian_random_matrix(n_components, n_features, random_state=None):
    """Generate a dense Gaussian random matrix.

    The components of the random matrix are drawn from

        N(0, 1.0 / n_components).

    Read more in the :ref:`User Guide <gaussian_random_matrix>`.

    Parameters
    ----------
    n_components : int,
        Dimensionality of the target projection space.

    n_features : int,
        Dimensionality of the original source space.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the matrix
        at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    components : ndarray of shape (n_components, n_features)
        The generated Gaussian random matrix.

    See Also
    --------
    GaussianRandomProjection
    """
    _check_input_size(n_components, n_features)
    rng = check_random_state(random_state)
    components = rng.normal(loc=0.0, scale=1.0 / np.sqrt(n_components), size=(n_components, n_features))
    return components