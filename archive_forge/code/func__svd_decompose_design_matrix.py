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
def _svd_decompose_design_matrix(self, X, y, sqrt_sw):
    X_mean = np.zeros(X.shape[1], dtype=X.dtype)
    if self.fit_intercept:
        intercept_column = sqrt_sw[:, None]
        X = np.hstack((X, intercept_column))
    U, singvals, _ = linalg.svd(X, full_matrices=0)
    singvals_sq = singvals ** 2
    UT_y = np.dot(U.T, y)
    return (X_mean, singvals_sq, U, UT_y)