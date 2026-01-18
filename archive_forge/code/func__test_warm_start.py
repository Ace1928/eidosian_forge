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
def _test_warm_start(klass, X, Y, lr):
    clf = klass(alpha=0.01, eta0=0.01, shuffle=False, learning_rate=lr)
    clf.fit(X, Y)
    clf2 = klass(alpha=0.001, eta0=0.01, shuffle=False, learning_rate=lr)
    clf2.fit(X, Y, coef_init=clf.coef_.copy(), intercept_init=clf.intercept_.copy())
    clf3 = klass(alpha=0.01, eta0=0.01, shuffle=False, warm_start=True, learning_rate=lr)
    clf3.fit(X, Y)
    assert clf3.t_ == clf.t_
    assert_array_almost_equal(clf3.coef_, clf.coef_)
    clf3.set_params(alpha=0.001)
    clf3.fit(X, Y)
    assert clf3.t_ == clf2.t_
    assert_array_almost_equal(clf3.coef_, clf2.coef_)