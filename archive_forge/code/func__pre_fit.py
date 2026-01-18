import numbers
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from scipy import linalg, optimize, sparse
from scipy.sparse.linalg import lsqr
from scipy.special import expit
from ..base import (
from ..utils import check_array, check_random_state
from ..utils._array_api import get_namespace
from ..utils._seq_dataset import (
from ..utils.extmath import safe_sparse_dot
from ..utils.parallel import Parallel, delayed
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import FLOAT_DTYPES, _check_sample_weight, check_is_fitted
def _pre_fit(X, y, Xy, precompute, fit_intercept, copy, check_input=True, sample_weight=None):
    """Function used at beginning of fit in linear models with L1 or L0 penalty.

    This function applies _preprocess_data and additionally computes the gram matrix
    `precompute` as needed as well as `Xy`.
    """
    n_samples, n_features = X.shape
    if sparse.issparse(X):
        precompute = False
        X, y, X_offset, y_offset, X_scale = _preprocess_data(X, y, fit_intercept=fit_intercept, copy=False, check_input=check_input, sample_weight=sample_weight)
    else:
        X, y, X_offset, y_offset, X_scale = _preprocess_data(X, y, fit_intercept=fit_intercept, copy=copy, check_input=check_input, sample_weight=sample_weight)
        if sample_weight is not None:
            X, y, _ = _rescale_data(X, y, sample_weight=sample_weight)
    if hasattr(precompute, '__array__'):
        if fit_intercept and (not np.allclose(X_offset, np.zeros(n_features))):
            warnings.warn('Gram matrix was provided but X was centered to fit intercept: recomputing Gram matrix.', UserWarning)
            precompute = 'auto'
            Xy = None
        elif check_input:
            _check_precomputed_gram_matrix(X, precompute, X_offset, X_scale)
    if isinstance(precompute, str) and precompute == 'auto':
        precompute = n_samples > n_features
    if precompute is True:
        precompute = np.empty(shape=(n_features, n_features), dtype=X.dtype, order='C')
        np.dot(X.T, X, out=precompute)
    if not hasattr(precompute, '__array__'):
        Xy = None
    if hasattr(precompute, '__array__') and Xy is None:
        common_dtype = np.result_type(X.dtype, y.dtype)
        if y.ndim == 1:
            Xy = np.empty(shape=n_features, dtype=common_dtype, order='C')
            np.dot(X.T, y, out=Xy)
        else:
            n_targets = y.shape[1]
            Xy = np.empty(shape=(n_features, n_targets), dtype=common_dtype, order='F')
            np.dot(y.T, X, out=Xy.T)
    return (X, y, X_offset, y_offset, X_scale, precompute, Xy)