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
Fit MultiTaskLasso model with coordinate descent.

        Fit is on grid of alphas and best alpha estimated by cross-validation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data.
        y : ndarray of shape (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary.

        **params : dict, default=None
            Parameters to be passed to the CV splitter.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Returns an instance of fitted model.
        