import atexit
import contextlib
import functools
import importlib
import inspect
import os
import os.path as op
import re
import shutil
import sys
import tempfile
import unittest
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from subprocess import STDOUT, CalledProcessError, TimeoutExpired, check_output
from unittest import TestCase
import joblib
import numpy as np
import scipy as sp
from numpy.testing import assert_allclose as np_assert_allclose
from numpy.testing import (
import sklearn
from sklearn.utils import (
from sklearn.utils._array_api import _check_array_api_dispatch
from sklearn.utils.fixes import VisibleDeprecationWarning, parse_version, sp_version
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
class MinimalTransformer:
    """Minimal transformer implementation with inheriting from
    BaseEstimator.

    This estimator should be tested with:

    * `check_estimator` in `test_estimator_checks.py`;
    * within a `Pipeline` in `test_pipeline.py`;
    * within a `SearchCV` in `test_search.py`.
    """

    def __init__(self, param=None):
        self.param = param

    def get_params(self, deep=True):
        return {'param': self.param}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y=None):
        check_array(X)
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = check_array(X)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)