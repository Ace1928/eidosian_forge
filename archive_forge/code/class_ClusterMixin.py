import copy
import functools
import inspect
import platform
import re
import warnings
from collections import defaultdict
import numpy as np
from . import __version__
from ._config import config_context, get_config
from .exceptions import InconsistentVersionWarning
from .utils import _IS_32BIT
from .utils._estimator_html_repr import _HTMLDocumentationLinkMixin, estimator_html_repr
from .utils._metadata_requests import _MetadataRequester, _routing_enabled
from .utils._param_validation import validate_parameter_constraints
from .utils._set_output import _SetOutputMixin
from .utils._tags import (
from .utils.validation import (
class ClusterMixin:
    """Mixin class for all cluster estimators in scikit-learn.

    - `_estimator_type` class attribute defaulting to `"clusterer"`;
    - `fit_predict` method returning the cluster labels associated to each sample.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator, ClusterMixin
    >>> class MyClusterer(ClusterMixin, BaseEstimator):
    ...     def fit(self, X, y=None):
    ...         self.labels_ = np.ones(shape=(len(X),), dtype=np.int64)
    ...         return self
    >>> X = [[1, 2], [2, 3], [3, 4]]
    >>> MyClusterer().fit_predict(X)
    array([1, 1, 1])
    """
    _estimator_type = 'clusterer'

    def fit_predict(self, X, y=None, **kwargs):
        """
        Perform clustering on `X` and returns cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : Ignored
            Not used, present for API consistency by convention.

        **kwargs : dict
            Arguments to be passed to ``fit``.

            .. versionadded:: 1.4

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype=np.int64
            Cluster labels.
        """
        self.fit(X, **kwargs)
        return self.labels_

    def _more_tags(self):
        return {'preserves_dtype': []}