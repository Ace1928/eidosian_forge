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
def _allocate_parameter_mem(self, n_classes, n_features, input_dtype, coef_init=None, intercept_init=None, one_class=0):
    """Allocate mem for parameters; initialize if provided."""
    if n_classes > 2:
        if coef_init is not None:
            coef_init = np.asarray(coef_init, dtype=input_dtype, order='C')
            if coef_init.shape != (n_classes, n_features):
                raise ValueError('Provided ``coef_`` does not match dataset. ')
            self.coef_ = coef_init
        else:
            self.coef_ = np.zeros((n_classes, n_features), dtype=input_dtype, order='C')
        if intercept_init is not None:
            intercept_init = np.asarray(intercept_init, order='C', dtype=input_dtype)
            if intercept_init.shape != (n_classes,):
                raise ValueError('Provided intercept_init does not match dataset.')
            self.intercept_ = intercept_init
        else:
            self.intercept_ = np.zeros(n_classes, dtype=input_dtype, order='C')
    else:
        if coef_init is not None:
            coef_init = np.asarray(coef_init, dtype=input_dtype, order='C')
            coef_init = coef_init.ravel()
            if coef_init.shape != (n_features,):
                raise ValueError('Provided coef_init does not match dataset.')
            self.coef_ = coef_init
        else:
            self.coef_ = np.zeros(n_features, dtype=input_dtype, order='C')
        if intercept_init is not None:
            intercept_init = np.asarray(intercept_init, dtype=input_dtype)
            if intercept_init.shape != (1,) and intercept_init.shape != ():
                raise ValueError('Provided intercept_init does not match dataset.')
            if one_class:
                self.offset_ = intercept_init.reshape(1)
            else:
                self.intercept_ = intercept_init.reshape(1)
        elif one_class:
            self.offset_ = np.zeros(1, dtype=input_dtype, order='C')
        else:
            self.intercept_ = np.zeros(1, dtype=input_dtype, order='C')
    if self.average > 0:
        self._standard_coef = self.coef_
        self._average_coef = np.zeros(self.coef_.shape, dtype=input_dtype, order='C')
        if one_class:
            self._standard_intercept = 1 - self.offset_
        else:
            self._standard_intercept = self.intercept_
        self._average_intercept = np.zeros(self._standard_intercept.shape, dtype=input_dtype, order='C')