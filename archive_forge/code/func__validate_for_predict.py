import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, ClassifierMixin, _fit_context
from ..exceptions import ConvergenceWarning, NotFittedError
from ..preprocessing import LabelEncoder
from ..utils import check_array, check_random_state, column_or_1d, compute_class_weight
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _ovr_decision_function, check_classification_targets
from ..utils.validation import (
from . import _liblinear as liblinear  # type: ignore
from . import _libsvm as libsvm  # type: ignore
from . import _libsvm_sparse as libsvm_sparse  # type: ignore
def _validate_for_predict(self, X):
    check_is_fitted(self)
    if not callable(self.kernel):
        X = self._validate_data(X, accept_sparse='csr', dtype=np.float64, order='C', accept_large_sparse=False, reset=False)
    if self._sparse and (not sp.issparse(X)):
        X = sp.csr_matrix(X)
    if self._sparse:
        X.sort_indices()
    if sp.issparse(X) and (not self._sparse) and (not callable(self.kernel)):
        raise ValueError('cannot use sparse input in %r trained on dense data' % type(self).__name__)
    if self.kernel == 'precomputed':
        if X.shape[1] != self.shape_fit_[0]:
            raise ValueError('X.shape[1] = %d should be equal to %d, the number of samples at training time' % (X.shape[1], self.shape_fit_[0]))
    sv = self.support_vectors_
    if not self._sparse and sv.size > 0 and (self.n_support_.sum() != sv.shape[0]):
        raise ValueError(f'The internal representation of {self.__class__.__name__} was altered')
    return X