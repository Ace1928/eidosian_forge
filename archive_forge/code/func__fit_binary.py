import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from ..base import (
from ..exceptions import ConvergenceWarning
from ..model_selection import ShuffleSplit, StratifiedShuffleSplit
from ..utils import check_random_state, compute_class_weight, deprecated
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _check_partial_fit_first_call
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, SparseCoefMixin, make_dataset
from ._sgd_fast import (
def _fit_binary(self, X, y, alpha, C, sample_weight, learning_rate, max_iter):
    """Fit a binary classifier on X and y."""
    coef, intercept, n_iter_ = fit_binary(self, 1, X, y, alpha, C, learning_rate, max_iter, self._expanded_class_weight[1], self._expanded_class_weight[0], sample_weight, random_state=self.random_state)
    self.t_ += n_iter_ * X.shape[0]
    self.n_iter_ = n_iter_
    if self.average > 0:
        if self.average <= self.t_ - 1:
            self.coef_ = self._average_coef.reshape(1, -1)
            self.intercept_ = self._average_intercept
        else:
            self.coef_ = self._standard_coef.reshape(1, -1)
            self._standard_intercept = np.atleast_1d(intercept)
            self.intercept_ = self._standard_intercept
    else:
        self.coef_ = coef.reshape(1, -1)
        self.intercept_ = np.atleast_1d(intercept)