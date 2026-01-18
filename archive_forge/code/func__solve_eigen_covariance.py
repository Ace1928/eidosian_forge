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
def _solve_eigen_covariance(self, alpha, y, sqrt_sw, X_mean, eigvals, V, X):
    """Compute dual coefficients and diagonal of G^-1.

        Used when we have a decomposition of X^T.X
        (n_samples > n_features and X is sparse).
        """
    if self.fit_intercept:
        return self._solve_eigen_covariance_intercept(alpha, y, sqrt_sw, X_mean, eigvals, V, X)
    return self._solve_eigen_covariance_no_intercept(alpha, y, sqrt_sw, X_mean, eigvals, V, X)