import itertools
import time
import warnings
from abc import ABC
from math import sqrt
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from .._config import config_context
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_random_state, gen_batches, metadata_routing
from ..utils._param_validation import (
from ..utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from ..utils.validation import (
from ._cdnmf_fast import _update_cdnmf_fast
def _check_w_h(self, X, W, H, update_H):
    """Check W and H, or initialize them."""
    n_samples, n_features = X.shape
    if self.init == 'custom' and update_H:
        _check_init(H, (self._n_components, n_features), 'NMF (input H)')
        _check_init(W, (n_samples, self._n_components), 'NMF (input W)')
        if self._n_components == 'auto':
            self._n_components = H.shape[0]
        if H.dtype != X.dtype or W.dtype != X.dtype:
            raise TypeError('H and W should have the same dtype as X. Got H.dtype = {} and W.dtype = {}.'.format(H.dtype, W.dtype))
    elif not update_H:
        if W is not None:
            warnings.warn('When update_H=False, the provided initial W is not used.', RuntimeWarning)
        _check_init(H, (self._n_components, n_features), 'NMF (input H)')
        if self._n_components == 'auto':
            self._n_components = H.shape[0]
        if H.dtype != X.dtype:
            raise TypeError('H should have the same dtype as X. Got H.dtype = {}.'.format(H.dtype))
        if self.solver == 'mu':
            avg = np.sqrt(X.mean() / self._n_components)
            W = np.full((n_samples, self._n_components), avg, dtype=X.dtype)
        else:
            W = np.zeros((n_samples, self._n_components), dtype=X.dtype)
    else:
        if W is not None or H is not None:
            warnings.warn("When init!='custom', provided W or H are ignored. Set  init='custom' to use them as initialization.", RuntimeWarning)
        if self._n_components == 'auto':
            self._n_components = X.shape[1]
        W, H = _initialize_nmf(X, self._n_components, init=self.init, random_state=self.random_state)
    return (W, H)