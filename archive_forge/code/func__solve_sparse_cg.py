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
def _solve_sparse_cg(X, y, alpha, max_iter=None, tol=0.0001, verbose=0, X_offset=None, X_scale=None, sample_weight_sqrt=None):
    if sample_weight_sqrt is None:
        sample_weight_sqrt = np.ones(X.shape[0], dtype=X.dtype)
    n_samples, n_features = X.shape
    if X_offset is None or X_scale is None:
        X1 = sp_linalg.aslinearoperator(X)
    else:
        X_offset_scale = X_offset / X_scale
        X1 = _get_rescaled_operator(X, X_offset_scale, sample_weight_sqrt)
    coefs = np.empty((y.shape[1], n_features), dtype=X.dtype)
    if n_features > n_samples:

        def create_mv(curr_alpha):

            def _mv(x):
                return X1.matvec(X1.rmatvec(x)) + curr_alpha * x
            return _mv
    else:

        def create_mv(curr_alpha):

            def _mv(x):
                return X1.rmatvec(X1.matvec(x)) + curr_alpha * x
            return _mv
    for i in range(y.shape[1]):
        y_column = y[:, i]
        mv = create_mv(alpha[i])
        if n_features > n_samples:
            C = sp_linalg.LinearOperator((n_samples, n_samples), matvec=mv, dtype=X.dtype)
            coef, info = _sparse_linalg_cg(C, y_column, rtol=tol)
            coefs[i] = X1.rmatvec(coef)
        else:
            y_column = X1.rmatvec(y_column)
            C = sp_linalg.LinearOperator((n_features, n_features), matvec=mv, dtype=X.dtype)
            coefs[i], info = _sparse_linalg_cg(C, y_column, maxiter=max_iter, rtol=tol)
        if info < 0:
            raise ValueError('Failed with error code %d' % info)
        if max_iter is None and info > 0 and verbose:
            warnings.warn('sparse_cg did not converge after %d iterations.' % info, ConvergenceWarning)
    return coefs