import numpy as np
from scipy import linalg
from ..utils import check_array
from ..utils._param_validation import StrOptions
from ..utils.extmath import row_norms
from ._base import BaseMixture, _check_shape
def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), 'means')
    return means