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
class _BaseRidge(LinearModel, metaclass=ABCMeta):
    _parameter_constraints: dict = {'alpha': [Interval(Real, 0, None, closed='left'), np.ndarray], 'fit_intercept': ['boolean'], 'copy_X': ['boolean'], 'max_iter': [Interval(Integral, 1, None, closed='left'), None], 'tol': [Interval(Real, 0, None, closed='left')], 'solver': [StrOptions({'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'})], 'positive': ['boolean'], 'random_state': ['random_state']}

    @abstractmethod
    def __init__(self, alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.positive = positive
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        if self.solver == 'lbfgs' and (not self.positive):
            raise ValueError("'lbfgs' solver can be used only when positive=True. Please use another solver.")
        if self.positive:
            if self.solver not in ['auto', 'lbfgs']:
                raise ValueError(f"solver='{self.solver}' does not support positive fitting. Please set the solver to 'auto' or 'lbfgs', or set `positive=False`")
            else:
                solver = self.solver
        elif sparse.issparse(X) and self.fit_intercept:
            if self.solver not in ['auto', 'lbfgs', 'lsqr', 'sag', 'sparse_cg']:
                raise ValueError("solver='{}' does not support fitting the intercept on sparse data. Please set the solver to 'auto' or 'lsqr', 'sparse_cg', 'sag', 'lbfgs' or set `fit_intercept=False`".format(self.solver))
            if self.solver in ['lsqr', 'lbfgs']:
                solver = self.solver
            elif self.solver == 'sag' and self.max_iter is None and (self.tol > 0.0001):
                warnings.warn('"sag" solver requires many iterations to fit an intercept with sparse inputs. Either set the solver to "auto" or "sparse_cg", or set a low "tol" and a high "max_iter" (especially if inputs are not standardized).')
                solver = 'sag'
            else:
                solver = 'sparse_cg'
        else:
            solver = self.solver
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        X, y, X_offset, y_offset, X_scale = _preprocess_data(X, y, fit_intercept=self.fit_intercept, copy=self.copy_X, sample_weight=sample_weight)
        if solver == 'sag' and sparse.issparse(X) and self.fit_intercept:
            self.coef_, self.n_iter_, self.intercept_ = _ridge_regression(X, y, alpha=self.alpha, sample_weight=sample_weight, max_iter=self.max_iter, tol=self.tol, solver='sag', positive=self.positive, random_state=self.random_state, return_n_iter=True, return_intercept=True, check_input=False)
            self.intercept_ += y_offset
        else:
            if sparse.issparse(X) and self.fit_intercept:
                params = {'X_offset': X_offset, 'X_scale': X_scale}
            else:
                params = {}
            self.coef_, self.n_iter_ = _ridge_regression(X, y, alpha=self.alpha, sample_weight=sample_weight, max_iter=self.max_iter, tol=self.tol, solver=solver, positive=self.positive, random_state=self.random_state, return_n_iter=True, return_intercept=False, check_input=False, fit_intercept=self.fit_intercept, **params)
            self._set_intercept(X_offset, y_offset, X_scale)
        return self