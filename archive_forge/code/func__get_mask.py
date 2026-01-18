from contextlib import suppress
import numpy as np
from scipy import sparse as sp
from . import is_scalar_nan
from .fixes import _object_dtype_isnan
def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == value_to_mask.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Input data, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    value_to_mask : {int, float}
        The value which is to be masked in X.

    Returns
    -------
    X_mask : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Missing mask.
    """
    if not sp.issparse(X):
        return _get_dense_mask(X, value_to_mask)
    Xt = _get_dense_mask(X.data, value_to_mask)
    sparse_constructor = sp.csr_matrix if X.format == 'csr' else sp.csc_matrix
    Xt_sparse = sparse_constructor((Xt, X.indices.copy(), X.indptr.copy()), shape=X.shape, dtype=bool)
    return Xt_sparse