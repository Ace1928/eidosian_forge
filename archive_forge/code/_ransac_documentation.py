import warnings
from numbers import Integral, Real
import numpy as np
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_consistent_length, check_random_state
from ..utils._param_validation import (
from ..utils.metadata_routing import (
from ..utils.random import sample_without_replacement
from ..utils.validation import _check_sample_weight, check_is_fitted, has_fit_parameter
from ._base import LinearRegression
Return the score of the prediction.

        This is a wrapper for `estimator_.score(X, y)`.

        Parameters
        ----------
        X : (array-like or sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        z : float
            Score of the prediction.
        