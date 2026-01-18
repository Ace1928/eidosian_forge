from numbers import Integral, Real
from time import time
import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import pdist, squareform
from ..base import (
from ..decomposition import PCA
from ..metrics.pairwise import _VALID_METRICS, pairwise_distances
from ..neighbors import NearestNeighbors
from ..utils import check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.validation import _num_samples, check_non_negative
from . import _barnes_hut_tsne, _utils  # type: ignore
def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded, neighbors=None, skip_num_points=0):
    """Runs t-SNE."""
    params = X_embedded.ravel()
    opt_args = {'it': 0, 'n_iter_check': self._N_ITER_CHECK, 'min_grad_norm': self.min_grad_norm, 'learning_rate': self.learning_rate_, 'verbose': self.verbose, 'kwargs': dict(skip_num_points=skip_num_points), 'args': [P, degrees_of_freedom, n_samples, self.n_components], 'n_iter_without_progress': self._EXPLORATION_N_ITER, 'n_iter': self._EXPLORATION_N_ITER, 'momentum': 0.5}
    if self.method == 'barnes_hut':
        obj_func = _kl_divergence_bh
        opt_args['kwargs']['angle'] = self.angle
        opt_args['kwargs']['verbose'] = self.verbose
        opt_args['kwargs']['num_threads'] = _openmp_effective_n_threads()
    else:
        obj_func = _kl_divergence
    P *= self.early_exaggeration
    params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)
    if self.verbose:
        print('[t-SNE] KL divergence after %d iterations with early exaggeration: %f' % (it + 1, kl_divergence))
    P /= self.early_exaggeration
    remaining = self.n_iter - self._EXPLORATION_N_ITER
    if it < self._EXPLORATION_N_ITER or remaining > 0:
        opt_args['n_iter'] = self.n_iter
        opt_args['it'] = it + 1
        opt_args['momentum'] = 0.8
        opt_args['n_iter_without_progress'] = self.n_iter_without_progress
        params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)
    self.n_iter_ = it
    if self.verbose:
        print('[t-SNE] KL divergence after %d iterations: %f' % (it + 1, kl_divergence))
    X_embedded = params.reshape(n_samples, self.n_components)
    self.kl_divergence_ = kl_divergence
    return X_embedded