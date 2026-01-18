from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
def _validate_matrix(self, A, name):
    A = np.atleast_2d(A)
    m, n = A.shape[-2:]
    if m != n or A.ndim != 2 or (not (np.issubdtype(A.dtype, np.integer) or np.issubdtype(A.dtype, np.floating))):
        message = f'The input `{name}` must be a square, two-dimensional array of real numbers.'
        raise ValueError(message)
    return A