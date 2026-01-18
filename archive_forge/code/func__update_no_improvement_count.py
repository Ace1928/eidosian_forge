import warnings
from abc import ABCMeta, abstractmethod
from itertools import chain
from numbers import Integral, Real
import numpy as np
import scipy.optimize
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics import accuracy_score, r2_score
from ..model_selection import train_test_split
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, Options, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import (
from ..utils.optimize import _check_optimize_result
from ..utils.validation import check_is_fitted
from ._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from ._stochastic_optimizers import AdamOptimizer, SGDOptimizer
def _update_no_improvement_count(self, early_stopping, X_val, y_val):
    if early_stopping:
        self.validation_scores_.append(self._score(X_val, y_val))
        if self.verbose:
            print('Validation score: %f' % self.validation_scores_[-1])
        last_valid_score = self.validation_scores_[-1]
        if last_valid_score < self.best_validation_score_ + self.tol:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
        if last_valid_score > self.best_validation_score_:
            self.best_validation_score_ = last_valid_score
            self._best_coefs = [c.copy() for c in self.coefs_]
            self._best_intercepts = [i.copy() for i in self.intercepts_]
    else:
        if self.loss_curve_[-1] > self.best_loss_ - self.tol:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
        if self.loss_curve_[-1] < self.best_loss_:
            self.best_loss_ = self.loss_curve_[-1]