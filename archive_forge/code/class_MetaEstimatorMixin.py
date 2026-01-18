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
class MetaEstimatorMixin:
    """Mixin class for all meta estimators in scikit-learn.

    This mixin defines the following functionality:

    - define `_required_parameters` that specify the mandatory `estimator` parameter.

    Examples
    --------
    >>> from sklearn.base import MetaEstimatorMixin
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> class MyEstimator(MetaEstimatorMixin):
    ...     def __init__(self, *, estimator=None):
    ...         self.estimator = estimator
    ...     def fit(self, X, y=None):
    ...         if self.estimator is None:
    ...             self.estimator_ = LogisticRegression()
    ...         else:
    ...             self.estimator_ = self.estimator
    ...         return self
    >>> X, y = load_iris(return_X_y=True)
    >>> estimator = MyEstimator().fit(X, y)
    >>> estimator.estimator_
    LogisticRegression()
    """
    _required_parameters = ['estimator']