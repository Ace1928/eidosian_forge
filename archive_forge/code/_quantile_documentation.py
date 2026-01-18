import warnings
from numbers import Real
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from ..base import BaseEstimator, RegressorMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..utils import _safe_indexing
from ..utils._param_validation import Interval, StrOptions
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import _check_sample_weight
from ._base import LinearModel
Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Returns self.
        