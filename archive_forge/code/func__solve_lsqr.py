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
def _solve_lsqr(X, y, *, alpha, fit_intercept=True, max_iter=None, tol=0.0001, X_offset=None, X_scale=None, sample_weight_sqrt=None):
    """Solve Ridge regression via LSQR.

    We expect that y is always mean centered.
    If X is dense, we expect it to be mean centered such that we can solve
        ||y - Xw||_2^2 + alpha * ||w||_2^2

    If X is sparse, we expect X_offset to be given such that we can solve
        ||y - (X - X_offset)w||_2^2 + alpha * ||w||_2^2

    With sample weights S=diag(sample_weight), this becomes
        ||sqrt(S) (y - (X - X_offset) w)||_2^2 + alpha * ||w||_2^2
    and we expect y and X to already be rescaled, i.e. sqrt(S) @ y, sqrt(S) @ X. In
    this case, X_offset is the sample_weight weighted mean of X before scaling by
    sqrt(S). The objective then reads
       ||y - (X - sqrt(S) X_offset) w)||_2^2 + alpha * ||w||_2^2
    """
    if sample_weight_sqrt is None:
        sample_weight_sqrt = np.ones(X.shape[0], dtype=X.dtype)
    if sparse.issparse(X) and fit_intercept:
        X_offset_scale = X_offset / X_scale
        X1 = _get_rescaled_operator(X, X_offset_scale, sample_weight_sqrt)
    else:
        X1 = X
    n_samples, n_features = X.shape
    coefs = np.empty((y.shape[1], n_features), dtype=X.dtype)
    n_iter = np.empty(y.shape[1], dtype=np.int32)
    sqrt_alpha = np.sqrt(alpha)
    for i in range(y.shape[1]):
        y_column = y[:, i]
        info = sp_linalg.lsqr(X1, y_column, damp=sqrt_alpha[i], atol=tol, btol=tol, iter_lim=max_iter)
        coefs[i] = info[0]
        n_iter[i] = info[2]
    return (coefs, n_iter)