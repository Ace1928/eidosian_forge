import warnings
import numpy as np
import scipy.sparse as sp
from ..base import _fit_context
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Integral, Interval, StrOptions
from ..utils.extmath import row_norms
from ..utils.validation import _check_sample_weight, check_is_fitted, check_random_state
from ._k_means_common import _inertia_dense, _inertia_sparse
from ._kmeans import (
def _warn_mkl_vcomp(self, n_active_threads):
    """Warn when vcomp and mkl are both present"""
    warnings.warn(f'BisectingKMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS={n_active_threads}.')