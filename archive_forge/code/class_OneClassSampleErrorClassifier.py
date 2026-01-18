import importlib
import sys
import unittest
import warnings
from numbers import Integral, Real
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn import config_context, get_config
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_multilabel_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning, SkipTestWarning
from sklearn.linear_model import (
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, NuSVC
from sklearn.utils import _array_api, all_estimators, deprecated
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
class OneClassSampleErrorClassifier(BaseBadClassifier):
    """Classifier allowing to trigger different behaviors when `sample_weight` reduces
    the number of classes to 1."""

    def __init__(self, raise_when_single_class=False):
        self.raise_when_single_class = raise_when_single_class

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=('csr', 'csc'), multi_output=True, y_numeric=True)
        self.has_single_class_ = False
        self.classes_, y = np.unique(y, return_inverse=True)
        n_classes_ = self.classes_.shape[0]
        if n_classes_ < 2 and self.raise_when_single_class:
            self.has_single_class_ = True
            raise ValueError('normal class error')
        if sample_weight is not None:
            if isinstance(sample_weight, np.ndarray) and len(sample_weight) > 0:
                n_classes_ = np.count_nonzero(np.bincount(y, sample_weight))
            if n_classes_ < 2:
                self.has_single_class_ = True
                raise ValueError('Nonsensical Error')
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if self.has_single_class_:
            return np.zeros(X.shape[0])
        return np.ones(X.shape[0])