import numpy as np
from scipy import linalg
from ..utils import check_array
from ..utils._param_validation import StrOptions
from ..utils.extmath import row_norms
from ._base import BaseMixture, _check_shape
def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like of shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(weights, (n_components,), 'weights')
    if any(np.less(weights, 0.0)) or any(np.greater(weights, 1.0)):
        raise ValueError("The parameter 'weights' should be in the range [0, 1], but got max value %.5f, min value %.5f" % (np.min(weights), np.max(weights)))
    if not np.allclose(np.abs(1.0 - np.sum(weights)), 0.0):
        raise ValueError("The parameter 'weights' should be normalized, but got sum(weights) = %.5f" % np.sum(weights))
    return weights