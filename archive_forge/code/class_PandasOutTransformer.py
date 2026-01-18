import pickle
import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
class PandasOutTransformer(BaseEstimator):

    def __init__(self, offset=1.0):
        self.offset = offset

    def fit(self, X, y=None):
        pd = pytest.importorskip('pandas')
        assert isinstance(X, pd.DataFrame)
        return self

    def transform(self, X, y=None):
        pd = pytest.importorskip('pandas')
        assert isinstance(X, pd.DataFrame)
        return X - self.offset

    def set_output(self, transform=None):
        return self