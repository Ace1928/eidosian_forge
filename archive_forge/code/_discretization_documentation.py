import warnings
from numbers import Integral
import numpy as np
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import _safe_indexing
from ..utils._param_validation import Hidden, Interval, Options, StrOptions
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
from ._encoders import OneHotEncoder
Get output feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        