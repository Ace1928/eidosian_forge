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
def _get_coef(self):
    if self.dual_coef_.shape[0] == 1:
        coef = safe_sparse_dot(self.dual_coef_, self.support_vectors_)
    else:
        coef = _one_vs_one_coef(self.dual_coef_, self._n_support, self.support_vectors_)
        if sp.issparse(coef[0]):
            coef = sp.vstack(coef).tocsr()
        else:
            coef = np.vstack(coef)
    return coef