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
class MockMetaEstimator:

    def __init__(self, delegate):
        """MetaEstimator to check if doctest on delegated methods work.

        Parameters
        ---------
        delegate : estimator
            Delegated estimator.
        """
        self.delegate = delegate

    @available_if(lambda self: hasattr(self.delegate, 'predict'))
    def predict(self, X):
        """This is available only if delegate has predict.

        Parameters
        ----------
        y : ndarray
            Parameter y
        """
        return self.delegate.predict(X)

    @available_if(lambda self: hasattr(self.delegate, 'score'))
    @deprecated('Testing a deprecated delegated method')
    def score(self, X):
        """This is available only if delegate has score.

        Parameters
        ---------
        y : ndarray
            Parameter y
        """

    @available_if(lambda self: hasattr(self.delegate, 'predict_proba'))
    def predict_proba(self, X):
        """This is available only if delegate has predict_proba.

        Parameters
        ---------
        X : ndarray
            Parameter X
        """
        return X

    @deprecated('Testing deprecated function with wrong params')
    def fit(self, X, y):
        """Incorrect docstring but should not be tested"""