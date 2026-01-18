import importlib
import inspect
import os
import warnings
from inspect import signature
from pkgutil import walk_packages
import numpy as np
import pytest
import sklearn
from sklearn.datasets import make_classification
from sklearn.experimental import (
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import IS_PYPY, all_estimators
from sklearn.utils._testing import (
from sklearn.utils.deprecation import _is_deprecated
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import parse_version, sp_version
def _get_all_fitted_attributes(estimator):
    """Get all the fitted attributes of an estimator including properties"""
    fit_attr = list(estimator.__dict__.keys())
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=FutureWarning)
        for name in dir(estimator.__class__):
            obj = getattr(estimator.__class__, name)
            if not isinstance(obj, property):
                continue
            try:
                getattr(estimator, name)
            except (AttributeError, FutureWarning):
                continue
            fit_attr.append(name)
    return [k for k in fit_attr if k.endswith('_') and (not k.startswith('_'))]