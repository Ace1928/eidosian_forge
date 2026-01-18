import numbers
import os
import pickle
import shutil
import tempfile
from copy import deepcopy
from functools import partial
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.linear_model import LogisticRegression, Perceptron, Ridge
from sklearn.metrics import (
from sklearn.metrics import cluster as cluster_module
from sklearn.metrics._scorer import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
from sklearn.utils.metadata_routing import MetadataRouter
def _make_estimators(X_train, y_train, y_ml_train):
    sensible_regr = DecisionTreeRegressor(random_state=0)
    sensible_regr.fit(X_train, _require_positive_y(y_train))
    sensible_clf = DecisionTreeClassifier(random_state=0)
    sensible_clf.fit(X_train, y_train)
    sensible_ml_clf = DecisionTreeClassifier(random_state=0)
    sensible_ml_clf.fit(X_train, y_ml_train)
    return dict([(name, sensible_regr) for name in REGRESSION_SCORERS] + [(name, sensible_clf) for name in CLF_SCORERS] + [(name, sensible_clf) for name in CLUSTER_SCORERS] + [(name, sensible_ml_clf) for name in MULTILABEL_ONLY_SCORERS])