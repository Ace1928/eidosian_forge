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
def _check_density(density, n_features):
    """Factorize density check according to Li et al."""
    if density == 'auto':
        density = 1 / np.sqrt(n_features)
    elif density <= 0 or density > 1:
        raise ValueError('Expected density in range ]0, 1], got: %r' % density)
    return density