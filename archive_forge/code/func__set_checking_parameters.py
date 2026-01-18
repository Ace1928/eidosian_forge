import pickle
import re
import warnings
from contextlib import nullcontext
from copy import deepcopy
from functools import partial, wraps
from inspect import signature
from numbers import Integral, Real
import joblib
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from .. import config_context
from ..base import (
from ..datasets import (
from ..exceptions import DataConversionWarning, NotFittedError, SkipTestWarning
from ..feature_selection import SelectFromModel, SelectKBest
from ..linear_model import (
from ..metrics import accuracy_score, adjusted_rand_score, f1_score
from ..metrics.pairwise import linear_kernel, pairwise_distances, rbf_kernel
from ..model_selection import ShuffleSplit, train_test_split
from ..model_selection._validation import _safe_split
from ..pipeline import make_pipeline
from ..preprocessing import StandardScaler, scale
from ..random_projection import BaseRandomProjection
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils._array_api import (
from ..utils._array_api import (
from ..utils._param_validation import (
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import check_is_fitted
from . import IS_PYPY, is_scalar_nan, shuffle
from ._param_validation import Interval
from ._tags import (
from ._testing import (
from .validation import _num_samples, has_fit_parameter
def _set_checking_parameters(estimator):
    params = estimator.get_params()
    name = estimator.__class__.__name__
    if name == 'TSNE':
        estimator.set_params(perplexity=2)
    if 'n_iter' in params and name != 'TSNE':
        estimator.set_params(n_iter=5)
    if 'max_iter' in params:
        if estimator.max_iter is not None:
            estimator.set_params(max_iter=min(5, estimator.max_iter))
        if name in ['LinearSVR', 'LinearSVC']:
            estimator.set_params(max_iter=20)
        if name == 'NMF':
            estimator.set_params(max_iter=500)
        if name == 'DictionaryLearning':
            estimator.set_params(max_iter=20, transform_algorithm='lasso_lars')
        if estimator.__class__.__name__ == 'MiniBatchNMF':
            estimator.set_params(max_iter=20, fresh_restarts=True)
        if name in ['MLPClassifier', 'MLPRegressor']:
            estimator.set_params(max_iter=100)
        if name == 'MiniBatchDictionaryLearning':
            estimator.set_params(max_iter=5)
    if 'n_resampling' in params:
        estimator.set_params(n_resampling=5)
    if 'n_estimators' in params:
        estimator.set_params(n_estimators=min(5, estimator.n_estimators))
    if 'max_trials' in params:
        estimator.set_params(max_trials=10)
    if 'n_init' in params:
        estimator.set_params(n_init=2)
    if 'batch_size' in params and (not name.startswith('MLP')):
        estimator.set_params(batch_size=10)
    if name == 'MeanShift':
        estimator.set_params(bandwidth=1.0)
    if name == 'TruncatedSVD':
        estimator.n_components = 1
    if name == 'LassoLarsIC':
        estimator.set_params(noise_variance=1.0)
    if hasattr(estimator, 'n_clusters'):
        estimator.n_clusters = min(estimator.n_clusters, 2)
    if hasattr(estimator, 'n_best'):
        estimator.n_best = 1
    if name == 'SelectFdr':
        estimator.set_params(alpha=0.5)
    if name == 'TheilSenRegressor':
        estimator.max_subpopulation = 100
    if isinstance(estimator, BaseRandomProjection):
        estimator.set_params(n_components=2)
    if isinstance(estimator, SelectKBest):
        estimator.set_params(k=1)
    if name in ('HistGradientBoostingClassifier', 'HistGradientBoostingRegressor'):
        estimator.set_params(min_samples_leaf=5)
    if name == 'DummyClassifier':
        estimator.set_params(strategy='stratified')
    loo_cv = ['RidgeCV', 'RidgeClassifierCV']
    if name not in loo_cv and hasattr(estimator, 'cv'):
        estimator.set_params(cv=3)
    if hasattr(estimator, 'n_splits'):
        estimator.set_params(n_splits=3)
    if name == 'OneHotEncoder':
        estimator.set_params(handle_unknown='ignore')
    if name == 'QuantileRegressor':
        solver = 'highs' if sp_version >= parse_version('1.6.0') else 'interior-point'
        estimator.set_params(solver=solver)
    if name in CROSS_DECOMPOSITION:
        estimator.set_params(n_components=1)
    if name == 'SpectralEmbedding':
        estimator.set_params(eigen_tol=1e-05)
    if name == 'HDBSCAN':
        estimator.set_params(min_samples=1)