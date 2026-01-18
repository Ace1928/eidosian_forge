import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def _compute_smoothed_state_weights(ssm, compute_t=None, compute_j=None, compute_prior_weights=None, scale=1.0):
    _model = ssm._statespace
    _kfilter = ssm._kalman_filter
    _smoother = ssm._kalman_smoother
    func = prefix_compute_smoothed_state_weights_map[ssm.prefix]
    if compute_t is None:
        compute_t = np.arange(ssm.nobs)
    if compute_j is None:
        compute_j = np.arange(ssm.nobs)
    compute_t = np.unique(np.atleast_1d(compute_t).astype(np.int32))
    compute_t.sort()
    compute_j = np.unique(np.atleast_1d(compute_j).astype(np.int32))
    compute_j.sort()
    if compute_prior_weights is None:
        compute_prior_weights = compute_j[0] == 0
    if compute_prior_weights and compute_j[0] != 0:
        raise ValueError('If `compute_prior_weights` is set to True, then `compute_j` must include the time period 0.')
    weights, state_intercept_weights, prior_weights, _ = func(_smoother, _kfilter, _model, compute_t, compute_j, scale, bool(compute_prior_weights))
    t0 = min(compute_t[0], compute_j[0])
    missing = np.isnan(ssm.endog[:, t0:])
    if np.any(missing):
        shape = weights.shape
        weights = np.asfortranarray(weights.transpose(2, 0, 1, 3).reshape(shape[2] * shape[0], shape[1], shape[3], order='C'))
        missing = np.asfortranarray(missing.astype(np.int32))
        reorder_missing_matrix(weights, missing, reorder_cols=True, inplace=True)
        weights = weights.reshape(shape[2], shape[0], shape[1], shape[3]).transpose(0, 3, 1, 2)
    else:
        weights = weights.transpose(2, 3, 0, 1)
    state_intercept_weights = state_intercept_weights.transpose(2, 3, 0, 1)
    prior_weights = prior_weights.transpose(2, 0, 1)
    ix_tj = np.ix_(compute_t - t0, compute_j - t0)
    weights = weights[ix_tj]
    state_intercept_weights = state_intercept_weights[ix_tj]
    if compute_prior_weights:
        prior_weights = prior_weights[compute_t - t0]
    return (weights, state_intercept_weights, prior_weights)