import itertools
import warnings
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import distance
from .. import config_context
from ..exceptions import DataConversionWarning
from ..preprocessing import normalize
from ..utils import (
from ..utils._mask import _get_mask
from ..utils._param_validation import (
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import parse_version, sp_base_version
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _num_samples, check_non_negative
from ._pairwise_distances_reduction import ArgKmin
from ._pairwise_fast import _chi2_kernel_fast, _sparse_manhattan
def _parallel_pairwise(X, Y, func, n_jobs, **kwds):
    """Break the pairwise matrix in n_jobs even slices
    and compute them in parallel."""
    if Y is None:
        Y = X
    X, Y, dtype = _return_float_dtype(X, Y)
    if effective_n_jobs(n_jobs) == 1:
        return func(X, Y, **kwds)
    fd = delayed(_dist_wrapper)
    ret = np.empty((X.shape[0], Y.shape[0]), dtype=dtype, order='F')
    Parallel(backend='threading', n_jobs=n_jobs)((fd(func, ret, s, X, Y[s], **kwds) for s in gen_even_slices(_num_samples(Y), effective_n_jobs(n_jobs))))
    if (X is Y or Y is None) and func is euclidean_distances:
        np.fill_diagonal(ret, 0)
    return ret