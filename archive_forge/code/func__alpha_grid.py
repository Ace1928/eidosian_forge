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
def _alpha_grid(X, y, Xy=None, l1_ratio=1.0, fit_intercept=True, eps=0.001, n_alphas=100, copy_X=True):
    """Compute the grid of alpha values for elastic net parameter search

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication

    y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Target values

    Xy : array-like of shape (n_features,) or (n_features, n_outputs),         default=None
        Xy = np.dot(X.T, y) that can be precomputed.

    l1_ratio : float, default=1.0
        The elastic net mixing parameter, with ``0 < l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. (currently not
        supported) ``For l1_ratio = 1`` it is an L1 penalty. For
        ``0 < l1_ratio <1``, the penalty is a combination of L1 and L2.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, default=100
        Number of alphas along the regularization path

    fit_intercept : bool, default=True
        Whether to fit an intercept or not

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.
    """
    if l1_ratio == 0:
        raise ValueError('Automatic alpha grid generation is not supported for l1_ratio=0. Please supply a grid by providing your estimator with the appropriate `alphas=` argument.')
    n_samples = len(y)
    sparse_center = False
    if Xy is None:
        X_sparse = sparse.issparse(X)
        sparse_center = X_sparse and fit_intercept
        X = check_array(X, accept_sparse='csc', copy=copy_X and fit_intercept and (not X_sparse))
        if not X_sparse:
            X, y, _, _, _ = _preprocess_data(X, y, fit_intercept=fit_intercept, copy=False)
        Xy = safe_sparse_dot(X.T, y, dense_output=True)
        if sparse_center:
            _, _, X_offset, _, X_scale = _preprocess_data(X, y, fit_intercept=fit_intercept)
            mean_dot = X_offset * np.sum(y)
    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]
    if sparse_center:
        if fit_intercept:
            Xy -= mean_dot[:, np.newaxis]
    alpha_max = np.sqrt(np.sum(Xy ** 2, axis=1)).max() / (n_samples * l1_ratio)
    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas
    return np.geomspace(alpha_max, alpha_max * eps, num=n_alphas)