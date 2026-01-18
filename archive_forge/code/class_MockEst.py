import atexit
import os
import unittest
import warnings
import numpy as np
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import (
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import available_if
class MockEst:

    def __init__(self):
        """MockEstimator"""

    def fit(self, X, y):
        return X

    def predict(self, X):
        return X

    def predict_proba(self, X):
        return X

    def score(self, X):
        return 1.0