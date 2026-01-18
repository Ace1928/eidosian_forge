import numbers
import sys
import warnings
from abc import ABC, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy import sparse
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..model_selection import check_cv
from ..utils import Bunch, check_array, check_scalar
from ..utils._metadata_requests import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import safe_sparse_dot
from ..utils.metadata_routing import (
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from . import _cd_fast as cd_fast  # type: ignore
from ._base import LinearModel, _pre_fit, _preprocess_data
def _path_residuals(X, y, sample_weight, train, test, fit_intercept, path, path_params, alphas=None, l1_ratio=1, X_order=None, dtype=None):
    """Returns the MSE for the models computed by 'path'.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    sample_weight : None or array-like of shape (n_samples,)
        Sample weights.

    train : list of indices
        The indices of the train set.

    test : list of indices
        The indices of the test set.

    path : callable
        Function returning a list of models on the path. See
        enet_path for an example of signature.

    path_params : dictionary
        Parameters passed to the path function.

    alphas : array-like, default=None
        Array of float that is used for cross-validation. If not
        provided, computed using 'path'.

    l1_ratio : float, default=1
        float between 0 and 1 passed to ElasticNet (scaling between
        l1 and l2 penalties). For ``l1_ratio = 0`` the penalty is an
        L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty. For ``0
        < l1_ratio < 1``, the penalty is a combination of L1 and L2.

    X_order : {'F', 'C'}, default=None
        The order of the arrays expected by the path function to
        avoid memory copies.

    dtype : a numpy dtype, default=None
        The dtype of the arrays expected by the path function to
        avoid memory copies.
    """
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
    if sample_weight is None:
        sw_train, sw_test = (None, None)
    else:
        sw_train = sample_weight[train]
        sw_test = sample_weight[test]
        n_samples = X_train.shape[0]
        sw_train *= n_samples / np.sum(sw_train)
    if not sparse.issparse(X):
        for array, array_input in ((X_train, X), (y_train, y), (X_test, X), (y_test, y)):
            if array.base is not array_input and (not array.flags['WRITEABLE']):
                array.setflags(write=True)
    if y.ndim == 1:
        precompute = path_params['precompute']
    else:
        precompute = False
    X_train, y_train, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(X_train, y_train, None, precompute, fit_intercept=fit_intercept, copy=False, sample_weight=sw_train)
    path_params = path_params.copy()
    path_params['Xy'] = Xy
    path_params['X_offset'] = X_offset
    path_params['X_scale'] = X_scale
    path_params['precompute'] = precompute
    path_params['copy_X'] = False
    path_params['alphas'] = alphas
    path_params['sample_weight'] = sw_train
    if 'l1_ratio' in path_params:
        path_params['l1_ratio'] = l1_ratio
    X_train = check_array(X_train, accept_sparse='csc', dtype=dtype, order=X_order)
    alphas, coefs, _ = path(X_train, y_train, **path_params)
    del X_train, y_train
    if y.ndim == 1:
        coefs = coefs[np.newaxis, :, :]
        y_offset = np.atleast_1d(y_offset)
        y_test = y_test[:, np.newaxis]
    intercepts = y_offset[:, np.newaxis] - np.dot(X_offset, coefs)
    X_test_coefs = safe_sparse_dot(X_test, coefs)
    residues = X_test_coefs - y_test[:, :, np.newaxis]
    residues += intercepts
    if sample_weight is None:
        this_mse = (residues ** 2).mean(axis=0)
    else:
        this_mse = np.average(residues ** 2, weights=sw_test, axis=0)
    return this_mse.mean(axis=0)