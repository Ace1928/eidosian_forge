import itertools
import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse
from ..base import BaseEstimator, MultiOutputMixin, is_classifier
from ..exceptions import DataConversionWarning, EfficiencyWarning
from ..metrics import DistanceMetric, pairwise_distances_chunked
from ..metrics._pairwise_distances_reduction import (
from ..metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from ..utils import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.fixes import parse_version, sp_base_version
from ..utils.multiclass import check_classification_targets
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted, check_non_negative
from ._ball_tree import BallTree
from ._kd_tree import KDTree
class NeighborsBase(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for nearest neighbors estimators."""
    _parameter_constraints: dict = {'n_neighbors': [Interval(Integral, 1, None, closed='left'), None], 'radius': [Interval(Real, 0, None, closed='both'), None], 'algorithm': [StrOptions({'auto', 'ball_tree', 'kd_tree', 'brute'})], 'leaf_size': [Interval(Integral, 1, None, closed='left')], 'p': [Interval(Real, 0, None, closed='right'), None], 'metric': [StrOptions(set(itertools.chain(*VALID_METRICS.values()))), callable], 'metric_params': [dict, None], 'n_jobs': [Integral, None]}

    @abstractmethod
    def __init__(self, n_neighbors=None, radius=None, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.n_jobs = n_jobs

    def _check_algorithm_metric(self):
        if self.algorithm == 'auto':
            if self.metric == 'precomputed':
                alg_check = 'brute'
            elif callable(self.metric) or self.metric in VALID_METRICS['ball_tree'] or isinstance(self.metric, DistanceMetric):
                alg_check = 'ball_tree'
            else:
                alg_check = 'brute'
        else:
            alg_check = self.algorithm
        if callable(self.metric):
            if self.algorithm == 'kd_tree':
                raise ValueError("kd_tree does not support callable metric '%s'Function call overhead will resultin very poor performance." % self.metric)
        elif self.metric not in VALID_METRICS[alg_check] and (not isinstance(self.metric, DistanceMetric)):
            raise ValueError("Metric '%s' not valid. Use sorted(sklearn.neighbors.VALID_METRICS['%s']) to get valid options. Metric can also be a callable function." % (self.metric, alg_check))
        if self.metric_params is not None and 'p' in self.metric_params:
            if self.p is not None:
                warnings.warn('Parameter p is found in metric_params. The corresponding parameter from __init__ is ignored.', SyntaxWarning, stacklevel=3)

    def _fit(self, X, y=None):
        if self._get_tags()['requires_y']:
            if not isinstance(X, (KDTree, BallTree, NeighborsBase)):
                X, y = self._validate_data(X, y, accept_sparse='csr', multi_output=True, order='C')
            if is_classifier(self):
                if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
                    if y.ndim != 1:
                        warnings.warn('A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().', DataConversionWarning, stacklevel=2)
                    self.outputs_2d_ = False
                    y = y.reshape((-1, 1))
                else:
                    self.outputs_2d_ = True
                check_classification_targets(y)
                self.classes_ = []
                self._y = np.empty(y.shape, dtype=np.intp)
                for k in range(self._y.shape[1]):
                    classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
                    self.classes_.append(classes)
                if not self.outputs_2d_:
                    self.classes_ = self.classes_[0]
                    self._y = self._y.ravel()
            else:
                self._y = y
        elif not isinstance(X, (KDTree, BallTree, NeighborsBase)):
            X = self._validate_data(X, accept_sparse='csr', order='C')
        self._check_algorithm_metric()
        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()
        effective_p = self.effective_metric_params_.get('p', self.p)
        if self.metric == 'minkowski':
            self.effective_metric_params_['p'] = effective_p
        self.effective_metric_ = self.metric
        if self.metric == 'minkowski':
            p = self.effective_metric_params_.pop('p', 2)
            w = self.effective_metric_params_.pop('w', None)
            if p == 1 and w is None:
                self.effective_metric_ = 'manhattan'
            elif p == 2 and w is None:
                self.effective_metric_ = 'euclidean'
            elif p == np.inf and w is None:
                self.effective_metric_ = 'chebyshev'
            else:
                self.effective_metric_params_['p'] = p
                self.effective_metric_params_['w'] = w
        if isinstance(X, NeighborsBase):
            self._fit_X = X._fit_X
            self._tree = X._tree
            self._fit_method = X._fit_method
            self.n_samples_fit_ = X.n_samples_fit_
            return self
        elif isinstance(X, BallTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = 'ball_tree'
            self.n_samples_fit_ = X.data.shape[0]
            return self
        elif isinstance(X, KDTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = 'kd_tree'
            self.n_samples_fit_ = X.data.shape[0]
            return self
        if self.metric == 'precomputed':
            X = _check_precomputed(X)
            if X.shape[0] != X.shape[1]:
                raise ValueError('Precomputed matrix must be square. Input is a {}x{} matrix.'.format(X.shape[0], X.shape[1]))
            self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError('n_samples must be greater than 0')
        if issparse(X):
            if self.algorithm not in ('auto', 'brute'):
                warnings.warn('cannot use tree with sparse input: using brute force')
            if self.effective_metric_ not in VALID_METRICS_SPARSE['brute'] and (not callable(self.effective_metric_)) and (not isinstance(self.effective_metric_, DistanceMetric)):
                raise ValueError("Metric '%s' not valid for sparse input. Use sorted(sklearn.neighbors.VALID_METRICS_SPARSE['brute']) to get valid options. Metric can also be a callable function." % self.effective_metric_)
            self._fit_X = X.copy()
            self._tree = None
            self._fit_method = 'brute'
            self.n_samples_fit_ = X.shape[0]
            return self
        self._fit_method = self.algorithm
        self._fit_X = X
        self.n_samples_fit_ = X.shape[0]
        if self._fit_method == 'auto':
            if self.metric == 'precomputed' or self._fit_X.shape[1] > 15 or (self.n_neighbors is not None and self.n_neighbors >= self._fit_X.shape[0] // 2):
                self._fit_method = 'brute'
            elif self.effective_metric_ == 'minkowski' and self.effective_metric_params_['p'] < 1:
                self._fit_method = 'brute'
            elif self.effective_metric_ == 'minkowski' and self.effective_metric_params_.get('w') is not None:
                self._fit_method = 'ball_tree'
            elif self.effective_metric_ in VALID_METRICS['kd_tree']:
                self._fit_method = 'kd_tree'
            elif callable(self.effective_metric_) or self.effective_metric_ in VALID_METRICS['ball_tree']:
                self._fit_method = 'ball_tree'
            else:
                self._fit_method = 'brute'
        if self.effective_metric_ == 'minkowski' and self.effective_metric_params_['p'] < 1:
            if self._fit_method == 'brute':
                warnings.warn("Mind that for 0 < p < 1, Minkowski metrics are not distance metrics. Continuing the execution with `algorithm='brute'`.")
            else:
                raise ValueError(f'algorithm="{self._fit_method}" does not support 0 < p < 1 for the Minkowski metric. To resolve this problem either set p >= 1 or algorithm="brute".')
        if self._fit_method == 'ball_tree':
            self._tree = BallTree(X, self.leaf_size, metric=self.effective_metric_, **self.effective_metric_params_)
        elif self._fit_method == 'kd_tree':
            if self.effective_metric_ == 'minkowski' and self.effective_metric_params_.get('w') is not None:
                raise ValueError("algorithm='kd_tree' is not valid for metric='minkowski' with a weight parameter 'w': try algorithm='ball_tree' or algorithm='brute' instead.")
            self._tree = KDTree(X, self.leaf_size, metric=self.effective_metric_, **self.effective_metric_params_)
        elif self._fit_method == 'brute':
            self._tree = None
        return self

    def _more_tags(self):
        return {'pairwise': self.metric == 'precomputed'}