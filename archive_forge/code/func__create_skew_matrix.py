import numpy as np
from scipy.linalg import solve_banded
from ._rotation import Rotation
def _create_skew_matrix(x):
    """Create skew-symmetric matrices corresponding to vectors.

    Parameters
    ----------
    x : ndarray, shape (n, 3)
        Set of vectors.

    Returns
    -------
    ndarray, shape (n, 3, 3)
    """
    result = np.zeros((len(x), 3, 3))
    result[:, 0, 1] = -x[:, 2]
    result[:, 0, 2] = x[:, 1]
    result[:, 1, 0] = x[:, 2]
    result[:, 1, 2] = -x[:, 0]
    result[:, 2, 0] = -x[:, 1]
    result[:, 2, 1] = x[:, 0]
    return result