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
def _solve_cholesky(X, y, alpha):
    n_features = X.shape[1]
    n_targets = y.shape[1]
    A = safe_sparse_dot(X.T, X, dense_output=True)
    Xy = safe_sparse_dot(X.T, y, dense_output=True)
    one_alpha = np.array_equal(alpha, len(alpha) * [alpha[0]])
    if one_alpha:
        A.flat[::n_features + 1] += alpha[0]
        return linalg.solve(A, Xy, assume_a='pos', overwrite_a=True).T
    else:
        coefs = np.empty([n_targets, n_features], dtype=X.dtype)
        for coef, target, current_alpha in zip(coefs, Xy.T, alpha):
            A.flat[::n_features + 1] += current_alpha
            coef[:] = linalg.solve(A, target, assume_a='pos', overwrite_a=False).ravel()
            A.flat[::n_features + 1] -= current_alpha
        return coefs