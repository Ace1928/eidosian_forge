import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from scipy import linalg, optimize, sparse
from scipy.sparse import linalg as sp_linalg
from ..base import MultiOutputMixin, RegressorMixin, _fit_context, is_classifier
from ..exceptions import ConvergenceWarning
from ..metrics import check_scoring, get_scorer_names
from ..model_selection import GridSearchCV
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import _sparse_linalg_cg
from ..utils.metadata_routing import (
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, LinearModel, _preprocess_data, _rescale_data
from ._sag import sag_solver
def _ridge_regression(X, y, alpha, sample_weight=None, solver='auto', max_iter=None, tol=0.0001, verbose=0, positive=False, random_state=None, return_n_iter=False, return_intercept=False, X_scale=None, X_offset=None, check_input=True, fit_intercept=False):
    has_sw = sample_weight is not None
    if solver == 'auto':
        if positive:
            solver = 'lbfgs'
        elif return_intercept:
            solver = 'sag'
        elif not sparse.issparse(X):
            solver = 'cholesky'
        else:
            solver = 'sparse_cg'
    if solver not in ('sparse_cg', 'cholesky', 'svd', 'lsqr', 'sag', 'saga', 'lbfgs'):
        raise ValueError("Known solvers are 'sparse_cg', 'cholesky', 'svd' 'lsqr', 'sag', 'saga' or 'lbfgs'. Got %s." % solver)
    if positive and solver != 'lbfgs':
        raise ValueError(f"When positive=True, only 'lbfgs' solver can be used. Please change solver {solver} to 'lbfgs' or set positive=False.")
    if solver == 'lbfgs' and (not positive):
        raise ValueError("'lbfgs' solver can be used only when positive=True. Please use another solver.")
    if return_intercept and solver != 'sag':
        raise ValueError("In Ridge, only 'sag' solver can directly fit the intercept. Please change solver to 'sag' or set return_intercept=False.")
    if check_input:
        _dtype = [np.float64, np.float32]
        _accept_sparse = _get_valid_accept_sparse(sparse.issparse(X), solver)
        X = check_array(X, accept_sparse=_accept_sparse, dtype=_dtype, order='C')
        y = check_array(y, dtype=X.dtype, ensure_2d=False, order=None)
    check_consistent_length(X, y)
    n_samples, n_features = X.shape
    if y.ndim > 2:
        raise ValueError('Target y has the wrong shape %s' % str(y.shape))
    ravel = False
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        ravel = True
    n_samples_, n_targets = y.shape
    if n_samples != n_samples_:
        raise ValueError('Number of samples in X and y does not correspond: %d != %d' % (n_samples, n_samples_))
    if has_sw:
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        if solver not in ['sag', 'saga']:
            X, y, sample_weight_sqrt = _rescale_data(X, y, sample_weight)
    if alpha is not None and (not isinstance(alpha, np.ndarray)):
        alpha = check_scalar(alpha, 'alpha', target_type=numbers.Real, min_val=0.0, include_boundaries='left')
    alpha = np.asarray(alpha, dtype=X.dtype).ravel()
    if alpha.size not in [1, n_targets]:
        raise ValueError('Number of targets and number of penalties do not correspond: %d != %d' % (alpha.size, n_targets))
    if alpha.size == 1 and n_targets > 1:
        alpha = np.repeat(alpha, n_targets)
    n_iter = None
    if solver == 'sparse_cg':
        coef = _solve_sparse_cg(X, y, alpha, max_iter=max_iter, tol=tol, verbose=verbose, X_offset=X_offset, X_scale=X_scale, sample_weight_sqrt=sample_weight_sqrt if has_sw else None)
    elif solver == 'lsqr':
        coef, n_iter = _solve_lsqr(X, y, alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, X_offset=X_offset, X_scale=X_scale, sample_weight_sqrt=sample_weight_sqrt if has_sw else None)
    elif solver == 'cholesky':
        if n_features > n_samples:
            K = safe_sparse_dot(X, X.T, dense_output=True)
            try:
                dual_coef = _solve_cholesky_kernel(K, y, alpha)
                coef = safe_sparse_dot(X.T, dual_coef, dense_output=True).T
            except linalg.LinAlgError:
                solver = 'svd'
        else:
            try:
                coef = _solve_cholesky(X, y, alpha)
            except linalg.LinAlgError:
                solver = 'svd'
    elif solver in ['sag', 'saga']:
        max_squared_sum = row_norms(X, squared=True).max()
        coef = np.empty((y.shape[1], n_features), dtype=X.dtype)
        n_iter = np.empty(y.shape[1], dtype=np.int32)
        intercept = np.zeros((y.shape[1],), dtype=X.dtype)
        for i, (alpha_i, target) in enumerate(zip(alpha, y.T)):
            init = {'coef': np.zeros((n_features + int(return_intercept), 1), dtype=X.dtype)}
            coef_, n_iter_, _ = sag_solver(X, target.ravel(), sample_weight, 'squared', alpha_i, 0, max_iter, tol, verbose, random_state, False, max_squared_sum, init, is_saga=solver == 'saga')
            if return_intercept:
                coef[i] = coef_[:-1]
                intercept[i] = coef_[-1]
            else:
                coef[i] = coef_
            n_iter[i] = n_iter_
        if intercept.shape[0] == 1:
            intercept = intercept[0]
        coef = np.asarray(coef)
    elif solver == 'lbfgs':
        coef = _solve_lbfgs(X, y, alpha, positive=positive, tol=tol, max_iter=max_iter, X_offset=X_offset, X_scale=X_scale, sample_weight_sqrt=sample_weight_sqrt if has_sw else None)
    if solver == 'svd':
        if sparse.issparse(X):
            raise TypeError('SVD solver does not support sparse inputs currently')
        coef = _solve_svd(X, y, alpha)
    if ravel:
        coef = coef.ravel()
    if return_n_iter and return_intercept:
        return (coef, n_iter, intercept)
    elif return_intercept:
        return (coef, intercept)
    elif return_n_iter:
        return (coef, n_iter)
    else:
        return coef