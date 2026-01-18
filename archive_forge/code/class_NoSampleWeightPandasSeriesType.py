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
class NoSampleWeightPandasSeriesType(BaseEstimator):

    def fit(self, X, y, sample_weight=None):
        X, y = self._validate_data(X, y, accept_sparse=('csr', 'csc'), multi_output=True, y_numeric=True)
        from pandas import Series
        if isinstance(sample_weight, Series):
            raise ValueError("Estimator does not accept 'sample_weight'of type pandas.Series")
        return self

    def predict(self, X):
        X = check_array(X)
        return np.ones(X.shape[0])