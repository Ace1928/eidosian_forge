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
def _solve_eigen_gram(self, alpha, y, sqrt_sw, X_mean, eigvals, Q, QT_y):
    """Compute dual coefficients and diagonal of G^-1.

        Used when we have a decomposition of X.X^T (n_samples <= n_features).
        """
    w = 1.0 / (eigvals + alpha)
    if self.fit_intercept:
        normalized_sw = sqrt_sw / np.linalg.norm(sqrt_sw)
        intercept_dim = _find_smallest_angle(normalized_sw, Q)
        w[intercept_dim] = 0
    c = np.dot(Q, self._diag_dot(w, QT_y))
    G_inverse_diag = self._decomp_diag(w, Q)
    if len(y.shape) != 1:
        G_inverse_diag = G_inverse_diag[:, np.newaxis]
    return (G_inverse_diag, c)