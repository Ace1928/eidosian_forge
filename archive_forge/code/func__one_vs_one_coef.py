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
def _one_vs_one_coef(dual_coef, n_support, support_vectors):
    """Generate primal coefficients from dual coefficients
    for the one-vs-one multi class LibSVM in the case
    of a linear kernel."""
    n_class = dual_coef.shape[0] + 1
    coef = []
    sv_locs = np.cumsum(np.hstack([[0], n_support]))
    for class1 in range(n_class):
        sv1 = support_vectors[sv_locs[class1]:sv_locs[class1 + 1], :]
        for class2 in range(class1 + 1, n_class):
            sv2 = support_vectors[sv_locs[class2]:sv_locs[class2 + 1], :]
            alpha1 = dual_coef[class2 - 1, sv_locs[class1]:sv_locs[class1 + 1]]
            alpha2 = dual_coef[class1, sv_locs[class2]:sv_locs[class2 + 1]]
            coef.append(safe_sparse_dot(alpha1, sv1) + safe_sparse_dot(alpha2, sv2))
    return coef