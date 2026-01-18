from numbers import Integral
import numpy as np
from joblib import effective_n_jobs
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..metrics import check_scoring
from ..model_selection import check_cv
from ..model_selection._validation import _score
from ..utils._param_validation import HasMethods, Interval, RealNotInt
from ..utils.metadata_routing import (
from ..utils.metaestimators import _safe_split, available_if
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
from ._base import SelectorMixin, _get_feature_importances
Fit the RFE model and automatically tune the number of selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the total number of features.

        y : array-like of shape (n_samples,)
            Target values (integers for classification, real numbers for
            regression).

        groups : array-like of shape (n_samples,) or None, default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Fitted estimator.
        