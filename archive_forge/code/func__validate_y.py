import itertools
import numbers
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral
from warnings import warn
import numpy as np
from ..base import ClassifierMixin, RegressorMixin, _fit_context
from ..metrics import accuracy_score, r2_score
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import check_random_state, column_or_1d, indices_to_mask
from ..utils._param_validation import HasMethods, Interval, RealNotInt
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.multiclass import check_classification_targets
from ..utils.parallel import Parallel, delayed
from ..utils.random import sample_without_replacement
from ..utils.validation import _check_sample_weight, check_is_fitted, has_fit_parameter
from ._base import BaseEnsemble, _partition_estimators
def _validate_y(self, y):
    y = column_or_1d(y, warn=True)
    check_classification_targets(y)
    self.classes_, y = np.unique(y, return_inverse=True)
    self.n_classes_ = len(self.classes_)
    return y