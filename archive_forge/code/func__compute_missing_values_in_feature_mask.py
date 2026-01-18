import copy
import numbers
from abc import ABCMeta, abstractmethod
from math import ceil
from numbers import Integral, Real
import numpy as np
from scipy.sparse import issparse
from ..base import (
from ..utils import Bunch, check_random_state, compute_sample_weight
from ..utils._param_validation import Hidden, Interval, RealNotInt, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import (
from . import _criterion, _splitter, _tree
from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import (
from ._utils import _any_isnan_axis0
def _compute_missing_values_in_feature_mask(self, X, estimator_name=None):
    """Return boolean mask denoting if there are missing values for each feature.

        This method also ensures that X is finite.

        Parameter
        ---------
        X : array-like of shape (n_samples, n_features), dtype=DOUBLE
            Input data.

        estimator_name : str or None, default=None
            Name to use when raising an error. Defaults to the class name.

        Returns
        -------
        missing_values_in_feature_mask : ndarray of shape (n_features,), or None
            Missing value mask. If missing values are not supported or there
            are no missing values, return None.
        """
    estimator_name = estimator_name or self.__class__.__name__
    common_kwargs = dict(estimator_name=estimator_name, input_name='X')
    if not self._support_missing_values(X):
        assert_all_finite(X, **common_kwargs)
        return None
    with np.errstate(over='ignore'):
        overall_sum = np.sum(X)
    if not np.isfinite(overall_sum):
        _assert_all_finite_element_wise(X, xp=np, allow_nan=True, **common_kwargs)
    if not np.isnan(overall_sum):
        return None
    missing_values_in_feature_mask = _any_isnan_axis0(X)
    return missing_values_in_feature_mask