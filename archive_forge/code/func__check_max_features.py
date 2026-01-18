from copy import deepcopy
from numbers import Integral, Real
import numpy as np
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone
from ..exceptions import NotFittedError
from ..utils._param_validation import HasMethods, Interval, Options
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.validation import _num_features, check_is_fitted, check_scalar
from ._base import SelectorMixin, _get_feature_importances
def _check_max_features(self, X):
    if self.max_features is not None:
        n_features = _num_features(X)
        if callable(self.max_features):
            max_features = self.max_features(X)
        else:
            max_features = self.max_features
        check_scalar(max_features, 'max_features', Integral, min_val=0, max_val=n_features)
        self.max_features_ = max_features