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
def _sparse_multidot_diag(self, X, A, X_mean, sqrt_sw):
    """Compute the diagonal of (X - X_mean).dot(A).dot((X - X_mean).T)
        without explicitly centering X nor computing X.dot(A)
        when X is sparse.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)

        A : ndarray of shape (n_features, n_features)

        X_mean : ndarray of shape (n_features,)

        sqrt_sw : ndarray of shape (n_features,)
            square roots of sample weights

        Returns
        -------
        diag : np.ndarray, shape (n_samples,)
            The computed diagonal.
        """
    intercept_col = scale = sqrt_sw
    batch_size = X.shape[1]
    diag = np.empty(X.shape[0], dtype=X.dtype)
    for start in range(0, X.shape[0], batch_size):
        batch = slice(start, min(X.shape[0], start + batch_size), 1)
        X_batch = np.empty((X[batch].shape[0], X.shape[1] + self.fit_intercept), dtype=X.dtype)
        if self.fit_intercept:
            X_batch[:, :-1] = X[batch].toarray() - X_mean * scale[batch][:, None]
            X_batch[:, -1] = intercept_col[batch]
        else:
            X_batch = X[batch].toarray()
        diag[batch] = (X_batch.dot(A) * X_batch).sum(axis=1)
    return diag