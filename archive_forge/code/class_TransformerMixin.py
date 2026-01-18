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
class TransformerMixin(_SetOutputMixin):
    """Mixin class for all transformers in scikit-learn.

    This mixin defines the following functionality:

    - a `fit_transform` method that delegates to `fit` and `transform`;
    - a `set_output` method to output `X` as a specific container type.

    If :term:`get_feature_names_out` is defined, then :class:`BaseEstimator` will
    automatically wrap `transform` and `fit_transform` to follow the `set_output`
    API. See the :ref:`developer_api_set_output` for details.

    :class:`OneToOneFeatureMixin` and
    :class:`ClassNamePrefixFeaturesOutMixin` are helpful mixins for
    defining :term:`get_feature_names_out`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator, TransformerMixin
    >>> class MyTransformer(TransformerMixin, BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y=None):
    ...         return self
    ...     def transform(self, X):
    ...         return np.full(shape=len(X), fill_value=self.param)
    >>> transformer = MyTransformer()
    >>> X = [[1, 2], [2, 3], [3, 4]]
    >>> transformer.fit_transform(X)
    array([1, 1, 1])
    """

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs),                 default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        if _routing_enabled():
            transform_params = self.get_metadata_routing().consumes(method='transform', params=fit_params.keys())
            if transform_params:
                warnings.warn(f"This object ({self.__class__.__name__}) has a `transform` method which consumes metadata, but `fit_transform` does not forward metadata to `transform`. Please implement a custom `fit_transform` method to forward metadata to `transform` as well. Alternatively, you can explicitly do `set_transform_request`and set all values to `False` to disable metadata routed to `transform`, if that's an option.", UserWarning)
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X)