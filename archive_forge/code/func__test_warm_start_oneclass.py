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
@pytest.mark.parametrize('klass', [SGDOneClassSVM, SparseSGDOneClassSVM])
def _test_warm_start_oneclass(klass, X, lr):
    clf = klass(nu=0.5, eta0=0.01, shuffle=False, learning_rate=lr)
    clf.fit(X)
    clf2 = klass(nu=0.1, eta0=0.01, shuffle=False, learning_rate=lr)
    clf2.fit(X, coef_init=clf.coef_.copy(), offset_init=clf.offset_.copy())
    clf3 = klass(nu=0.5, eta0=0.01, shuffle=False, warm_start=True, learning_rate=lr)
    clf3.fit(X)
    assert clf3.t_ == clf.t_
    assert_allclose(clf3.coef_, clf.coef_)
    clf3.set_params(nu=0.1)
    clf3.fit(X)
    assert clf3.t_ == clf2.t_
    assert_allclose(clf3.coef_, clf2.coef_)