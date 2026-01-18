import pickle
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import datasets, linear_model, metrics
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import _sgd_fast as sgd_fast
from sklearn.linear_model import _stochastic_gradient
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, scale
from sklearn.svm import OneClassSVM
from sklearn.utils._testing import (
class _SparseSGDClassifier(linear_model.SGDClassifier):

    def fit(self, X, y, *args, **kw):
        X = sp.csr_matrix(X)
        return super().fit(X, y, *args, **kw)

    def partial_fit(self, X, y, *args, **kw):
        X = sp.csr_matrix(X)
        return super().partial_fit(X, y, *args, **kw)

    def decision_function(self, X):
        X = sp.csr_matrix(X)
        return super().decision_function(X)

    def predict_proba(self, X):
        X = sp.csr_matrix(X)
        return super().predict_proba(X)