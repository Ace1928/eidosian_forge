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
class UntaggedBinaryClassifier(SGDClassifier):

    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        super().fit(X, y, coef_init, intercept_init, sample_weight)
        if len(self.classes_) > 2:
            raise ValueError('Only 2 classes are supported')
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        super().partial_fit(X=X, y=y, classes=classes, sample_weight=sample_weight)
        if len(self.classes_) > 2:
            raise ValueError('Only 2 classes are supported')
        return self