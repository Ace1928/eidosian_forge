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
def _partial_fit(self, X, alpha, C, loss, learning_rate, max_iter, sample_weight, coef_init, offset_init):
    first_call = getattr(self, 'coef_', None) is None
    X = self._validate_data(X, None, accept_sparse='csr', dtype=[np.float64, np.float32], order='C', accept_large_sparse=False, reset=first_call)
    n_features = X.shape[1]
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
    if getattr(self, 'coef_', None) is None or coef_init is not None:
        self._allocate_parameter_mem(n_classes=1, n_features=n_features, input_dtype=X.dtype, coef_init=coef_init, intercept_init=offset_init, one_class=1)
    elif n_features != self.coef_.shape[-1]:
        raise ValueError('Number of features %d does not match previous data %d.' % (n_features, self.coef_.shape[-1]))
    if self.average and getattr(self, '_average_coef', None) is None:
        self._average_coef = np.zeros(n_features, dtype=X.dtype, order='C')
        self._average_intercept = np.zeros(1, dtype=X.dtype, order='C')
    self._loss_function_ = self._get_loss_function(loss)
    if not hasattr(self, 't_'):
        self.t_ = 1.0
    self._fit_one_class(X, alpha=alpha, C=C, learning_rate=learning_rate, sample_weight=sample_weight, max_iter=max_iter)
    return self