import numpy as np
from scipy import linalg
from ..utils import check_array
from ..utils._param_validation import StrOptions
from ..utils.extmath import row_norms
from ._base import BaseMixture, _check_shape
def _check_precisions(precisions, covariance_type, n_components, n_features):
    """Validate user provided precisions.

    Parameters
    ----------
    precisions : array-like
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : str

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precisions = check_array(precisions, dtype=[np.float64, np.float32], ensure_2d=False, allow_nd=covariance_type == 'full')
    precisions_shape = {'full': (n_components, n_features, n_features), 'tied': (n_features, n_features), 'diag': (n_components, n_features), 'spherical': (n_components,)}
    _check_shape(precisions, precisions_shape[covariance_type], '%s precision' % covariance_type)
    _check_precisions = {'full': _check_precisions_full, 'tied': _check_precision_matrix, 'diag': _check_precision_positivity, 'spherical': _check_precision_positivity}
    _check_precisions[covariance_type](precisions, covariance_type)
    return precisions