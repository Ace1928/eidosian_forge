import numbers
import sys
import warnings
from abc import ABC, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy import sparse
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..model_selection import check_cv
from ..utils import Bunch, check_array, check_scalar
from ..utils._metadata_requests import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import safe_sparse_dot
from ..utils.metadata_routing import (
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from . import _cd_fast as cd_fast  # type: ignore
from ._base import LinearModel, _pre_fit, _preprocess_data
def _decision_function(self, X):
    """Decision function of the linear model.

        Parameters
        ----------
        X : numpy array or scipy.sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        T : ndarray of shape (n_samples,)
            The predicted decision function.
        """
    check_is_fitted(self)
    if sparse.issparse(X):
        return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    else:
        return super()._decision_function(X)