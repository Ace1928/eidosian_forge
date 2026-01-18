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
def _compute_gram(self, X, sqrt_sw):
    """Computes the Gram matrix XX^T with possible centering.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The preprocessed design matrix.

        sqrt_sw : ndarray of shape (n_samples,)
            square roots of sample weights

        Returns
        -------
        gram : ndarray of shape (n_samples, n_samples)
            The Gram matrix.
        X_mean : ndarray of shape (n_feature,)
            The weighted mean of ``X`` for each feature.

        Notes
        -----
        When X is dense the centering has been done in preprocessing
        so the mean is 0 and we just compute XX^T.

        When X is sparse it has not been centered in preprocessing, but it has
        been scaled by sqrt(sample weights).

        When self.fit_intercept is False no centering is done.

        The centered X is never actually computed because centering would break
        the sparsity of X.
        """
    center = self.fit_intercept and sparse.issparse(X)
    if not center:
        X_mean = np.zeros(X.shape[1], dtype=X.dtype)
        return (safe_sparse_dot(X, X.T, dense_output=True), X_mean)
    n_samples = X.shape[0]
    sample_weight_matrix = sparse.dia_matrix((sqrt_sw, 0), shape=(n_samples, n_samples))
    X_weighted = sample_weight_matrix.dot(X)
    X_mean, _ = mean_variance_axis(X_weighted, axis=0)
    X_mean *= n_samples / sqrt_sw.dot(sqrt_sw)
    X_mX = sqrt_sw[:, None] * safe_sparse_dot(X_mean, X.T, dense_output=True)
    X_mX_m = np.outer(sqrt_sw, sqrt_sw) * np.dot(X_mean, X_mean)
    return (safe_sparse_dot(X, X.T, dense_output=True) + X_mX_m - X_mX - X_mX.T, X_mean)