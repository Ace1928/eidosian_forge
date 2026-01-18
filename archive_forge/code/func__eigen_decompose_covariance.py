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
def _eigen_decompose_covariance(self, X, y, sqrt_sw):
    """Eigendecomposition of X^T.X, used when n_samples > n_features
        and X is sparse.
        """
    n_samples, n_features = X.shape
    cov = np.empty((n_features + 1, n_features + 1), dtype=X.dtype)
    cov[:-1, :-1], X_mean = self._compute_covariance(X, sqrt_sw)
    if not self.fit_intercept:
        cov = cov[:-1, :-1]
    else:
        cov[-1] = 0
        cov[:, -1] = 0
        cov[-1, -1] = sqrt_sw.dot(sqrt_sw)
    nullspace_dim = max(0, n_features - n_samples)
    eigvals, V = linalg.eigh(cov)
    eigvals = eigvals[nullspace_dim:]
    V = V[:, nullspace_dim:]
    return (X_mean, eigvals, V, X)