from functools import partial
import numpy as np
from sklearn.base import (
from sklearn.metrics._scorer import _Scorer, mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.multiclass import _check_partial_fit_first_call
class ConsumingTransformer(TransformerMixin, BaseEstimator):
    """A transformer which accepts metadata on fit and transform.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.
    """

    def __init__(self, registry=None):
        self.registry = registry

    def fit(self, X, y=None, sample_weight=None, metadata=None):
        if self.registry is not None:
            self.registry.append(self)
        record_metadata_not_default(self, 'fit', sample_weight=sample_weight, metadata=metadata)
        return self

    def transform(self, X, sample_weight=None, metadata=None):
        record_metadata(self, 'transform', sample_weight=sample_weight, metadata=metadata)
        return X

    def fit_transform(self, X, y, sample_weight=None, metadata=None):
        record_metadata(self, 'fit_transform', sample_weight=sample_weight, metadata=metadata)
        return self.fit(X, y, sample_weight=sample_weight, metadata=metadata).transform(X, sample_weight=sample_weight, metadata=metadata)

    def inverse_transform(self, X, sample_weight=None, metadata=None):
        record_metadata(self, 'inverse_transform', sample_weight=sample_weight, metadata=metadata)
        return X