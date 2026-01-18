import numbers
import warnings
from collections import Counter
from functools import partial
import numpy as np
import numpy.ma as ma
from scipy import sparse as sp
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import _is_pandas_na, is_scalar_nan
from ..utils._mask import _get_mask
from ..utils._param_validation import MissingValues, StrOptions
from ..utils.fixes import _mode
from ..utils.sparsefuncs import _get_median
from ..utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted
def _get_missing_features_info(self, X):
    """Compute the imputer mask and the indices of the features
        containing missing values.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data with missing values. Note that `X` has been
            checked in :meth:`fit` and :meth:`transform` before to call this
            function.

        Returns
        -------
        imputer_mask : {ndarray, sparse matrix} of shape         (n_samples, n_features)
            The imputer mask of the original data.

        features_with_missing : ndarray of shape (n_features_with_missing)
            The features containing missing values.
        """
    if not self._precomputed:
        imputer_mask = _get_mask(X, self.missing_values)
    else:
        imputer_mask = X
    if sp.issparse(X):
        imputer_mask.eliminate_zeros()
        if self.features == 'missing-only':
            n_missing = imputer_mask.getnnz(axis=0)
        if self.sparse is False:
            imputer_mask = imputer_mask.toarray()
        elif imputer_mask.format == 'csr':
            imputer_mask = imputer_mask.tocsc()
    else:
        if not self._precomputed:
            imputer_mask = _get_mask(X, self.missing_values)
        else:
            imputer_mask = X
        if self.features == 'missing-only':
            n_missing = imputer_mask.sum(axis=0)
        if self.sparse is True:
            imputer_mask = sp.csc_matrix(imputer_mask)
    if self.features == 'all':
        features_indices = np.arange(X.shape[1])
    else:
        features_indices = np.flatnonzero(n_missing)
    return (imputer_mask, features_indices)