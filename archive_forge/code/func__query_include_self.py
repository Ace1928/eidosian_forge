import itertools
from ..base import ClassNamePrefixFeaturesOutMixin, TransformerMixin, _fit_context
from ..utils._param_validation import (
from ..utils.validation import check_is_fitted
from ._base import VALID_METRICS, KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin
from ._unsupervised import NearestNeighbors
def _query_include_self(X, include_self, mode):
    """Return the query based on include_self param"""
    if include_self == 'auto':
        include_self = mode == 'connectivity'
    if not include_self:
        X = None
    return X