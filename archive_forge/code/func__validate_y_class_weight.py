import threading
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from warnings import catch_warnings, simplefilter, warn
import numpy as np
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse
from ..base import (
from ..exceptions import DataConversionWarning
from ..metrics import accuracy_score, r2_score
from ..preprocessing import OneHotEncoder
from ..tree import (
from ..tree._tree import DOUBLE, DTYPE
from ..utils import check_random_state, compute_sample_weight
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils._tags import _safe_tags
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import BaseEnsemble, _partition_estimators
def _validate_y_class_weight(self, y):
    check_classification_targets(y)
    y = np.copy(y)
    expanded_class_weight = None
    if self.class_weight is not None:
        y_original = np.copy(y)
    self.classes_ = []
    self.n_classes_ = []
    y_store_unique_indices = np.zeros(y.shape, dtype=int)
    for k in range(self.n_outputs_):
        classes_k, y_store_unique_indices[:, k] = np.unique(y[:, k], return_inverse=True)
        self.classes_.append(classes_k)
        self.n_classes_.append(classes_k.shape[0])
    y = y_store_unique_indices
    if self.class_weight is not None:
        valid_presets = ('balanced', 'balanced_subsample')
        if isinstance(self.class_weight, str):
            if self.class_weight not in valid_presets:
                raise ValueError('Valid presets for class_weight include "balanced" and "balanced_subsample".Given "%s".' % self.class_weight)
            if self.warm_start:
                warn('class_weight presets "balanced" or "balanced_subsample" are not recommended for warm_start if the fitted data differs from the full dataset. In order to use "balanced" weights, use compute_class_weight ("balanced", classes, y). In place of y you can use a large enough sample of the full training set target to properly estimate the class frequency distributions. Pass the resulting weights as the class_weight parameter.')
        if self.class_weight != 'balanced_subsample' or not self.bootstrap:
            if self.class_weight == 'balanced_subsample':
                class_weight = 'balanced'
            else:
                class_weight = self.class_weight
            expanded_class_weight = compute_sample_weight(class_weight, y_original)
    return (y, expanded_class_weight)