import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils._testing import assert_allclose, assert_no_warnings
class DummyCheckerArrayTransformer(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        assert isinstance(X, np.ndarray)
        return self

    def transform(self, X):
        assert isinstance(X, np.ndarray)
        return X

    def inverse_transform(self, X):
        assert isinstance(X, np.ndarray)
        return X