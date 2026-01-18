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
def _sparse_predict(self, X):
    kernel = self.kernel
    if callable(kernel):
        kernel = 'precomputed'
    kernel_type = self._sparse_kernels.index(kernel)
    C = 0.0
    return libsvm_sparse.libsvm_sparse_predict(X.data, X.indices, X.indptr, self.support_vectors_.data, self.support_vectors_.indices, self.support_vectors_.indptr, self._dual_coef_.data, self._intercept_, LIBSVM_IMPL.index(self._impl), kernel_type, self.degree, self._gamma, self.coef0, self.tol, C, getattr(self, 'class_weight_', np.empty(0)), self.nu, self.epsilon, self.shrinking, self.probability, self._n_support, self._probA, self._probB)