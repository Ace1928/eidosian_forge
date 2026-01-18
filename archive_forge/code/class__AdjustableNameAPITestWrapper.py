from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from sklearn._config import config_context
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
class _AdjustableNameAPITestWrapper(_ArrayAPIWrapper):
    """API wrapper that has an adjustable name. Used for testing."""

    def __init__(self, array_namespace, name):
        super().__init__(array_namespace=array_namespace)
        self.__name__ = name