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
def _check_test_data(self, X):
    X = self._validate_data(X, accept_sparse='csr', reset=False, dtype=[np.float64, np.float32], order='C', accept_large_sparse=False)
    return X