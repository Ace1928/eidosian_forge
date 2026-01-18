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
def _check_precomputed(X):
    """Check precomputed distance matrix.

    If the precomputed distance matrix is sparse, it checks that the non-zero
    entries are sorted by distances. If not, the matrix is copied and sorted.

    Parameters
    ----------
    X : {sparse matrix, array-like}, (n_samples, n_samples)
        Distance matrix to other samples. X may be a sparse matrix, in which
        case only non-zero elements may be considered neighbors.

    Returns
    -------
    X : {sparse matrix, array-like}, (n_samples, n_samples)
        Distance matrix to other samples. X may be a sparse matrix, in which
        case only non-zero elements may be considered neighbors.
    """
    if not issparse(X):
        X = check_array(X)
        check_non_negative(X, whom='precomputed distance matrix.')
        return X
    else:
        graph = X
    if graph.format not in ('csr', 'csc', 'coo', 'lil'):
        raise TypeError('Sparse matrix in {!r} format is not supported due to its handling of explicit zeros'.format(graph.format))
    copied = graph.format != 'csr'
    graph = check_array(graph, accept_sparse='csr')
    check_non_negative(graph, whom='precomputed distance matrix.')
    graph = sort_graph_by_row_values(graph, copy=not copied, warn_when_not_sorted=True)
    return graph