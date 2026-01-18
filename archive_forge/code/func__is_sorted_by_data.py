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
def _is_sorted_by_data(graph):
    """Return whether the graph's non-zero entries are sorted by data.

    The non-zero entries are stored in graph.data and graph.indices.
    For each row (or sample), the non-zero entries can be either:
        - sorted by indices, as after graph.sort_indices();
        - sorted by data, as after _check_precomputed(graph);
        - not sorted.

    Parameters
    ----------
    graph : sparse matrix of shape (n_samples, n_samples)
        Neighbors graph as given by `kneighbors_graph` or
        `radius_neighbors_graph`. Matrix should be of format CSR format.

    Returns
    -------
    res : bool
        Whether input graph is sorted by data.
    """
    assert graph.format == 'csr'
    out_of_order = graph.data[:-1] > graph.data[1:]
    line_change = np.unique(graph.indptr[1:-1] - 1)
    line_change = line_change[line_change < out_of_order.shape[0]]
    return out_of_order.sum() == out_of_order[line_change].sum()