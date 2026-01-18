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
def _precompute_metric_params(X, Y, metric=None, **kwds):
    """Precompute data-derived metric parameters if not provided."""
    if metric == 'seuclidean' and 'V' not in kwds:
        if X is Y:
            V = np.var(X, axis=0, ddof=1)
        else:
            raise ValueError("The 'V' parameter is required for the seuclidean metric when Y is passed.")
        return {'V': V}
    if metric == 'mahalanobis' and 'VI' not in kwds:
        if X is Y:
            VI = np.linalg.inv(np.cov(X.T)).T
        else:
            raise ValueError("The 'VI' parameter is required for the mahalanobis metric when Y is passed.")
        return {'VI': VI}
    return {}