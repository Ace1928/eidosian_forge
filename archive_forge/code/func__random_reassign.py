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
def _random_reassign(self):
    """Check if a random reassignment needs to be done.

        Do random reassignments each time 10 * n_clusters samples have been
        processed.

        If there are empty clusters we always want to reassign.
        """
    self._n_since_last_reassign += self._batch_size
    if (self._counts == 0).any() or self._n_since_last_reassign >= 10 * self.n_clusters:
        self._n_since_last_reassign = 0
        return True
    return False