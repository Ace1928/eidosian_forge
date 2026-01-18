import warnings
from collections import namedtuple
from numbers import Integral, Real
from time import time
import numpy as np
from scipy import stats
from ..base import _fit_context, clone
from ..exceptions import ConvergenceWarning
from ..preprocessing import normalize
from ..utils import (
from ..utils._mask import _get_mask
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.metadata_routing import _RoutingNotSupportedMixin
from ..utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted
from ._base import SimpleImputer, _BaseImputer, _check_inputs_dtype
@staticmethod
def _validate_limit(limit, limit_type, n_features):
    """Validate the limits (min/max) of the feature values.

        Converts scalar min/max limits to vectors of shape `(n_features,)`.

        Parameters
        ----------
        limit: scalar or array-like
            The user-specified limit (i.e, min_value or max_value).
        limit_type: {'max', 'min'}
            Type of limit to validate.
        n_features: int
            Number of features in the dataset.

        Returns
        -------
        limit: ndarray, shape(n_features,)
            Array of limits, one for each feature.
        """
    limit_bound = np.inf if limit_type == 'max' else -np.inf
    limit = limit_bound if limit is None else limit
    if np.isscalar(limit):
        limit = np.full(n_features, limit)
    limit = check_array(limit, force_all_finite=False, copy=False, ensure_2d=False)
    if not limit.shape[0] == n_features:
        raise ValueError(f"'{limit_type}_value' should be of shape ({n_features},) when an array-like is provided. Got {limit.shape}, instead.")
    return limit