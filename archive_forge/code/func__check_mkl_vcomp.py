import warnings
from abc import ABC, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import _euclidean_distances, euclidean_distances
from ..utils import check_array, check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, stable_cumsum
from ..utils.fixes import threadpool_info, threadpool_limits
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.sparsefuncs_fast import assign_rows_csr
from ..utils.validation import (
from ._k_means_common import (
from ._k_means_elkan import (
from ._k_means_lloyd import lloyd_iter_chunked_dense, lloyd_iter_chunked_sparse
from ._k_means_minibatch import _minibatch_update_dense, _minibatch_update_sparse
def _check_mkl_vcomp(self, X, n_samples):
    """Check when vcomp and mkl are both present"""
    if sp.issparse(X):
        return
    n_active_threads = int(np.ceil(n_samples / CHUNK_SIZE))
    if n_active_threads < self._n_threads:
        modules = threadpool_info()
        has_vcomp = 'vcomp' in [module['prefix'] for module in modules]
        has_mkl = ('mkl', 'intel') in [(module['internal_api'], module.get('threading_layer', None)) for module in modules]
        if has_vcomp and has_mkl:
            self._warn_mkl_vcomp(n_active_threads)