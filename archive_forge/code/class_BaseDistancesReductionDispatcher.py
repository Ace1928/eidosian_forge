from abc import abstractmethod
from typing import List
import numpy as np
from scipy.sparse import issparse
from ... import get_config
from .._dist_metrics import (
from ._argkmin import (
from ._argkmin_classmode import (
from ._base import _sqeuclidean_row_norms32, _sqeuclidean_row_norms64
from ._radius_neighbors import (
from ._radius_neighbors_classmode import (
class BaseDistancesReductionDispatcher:
    """Abstract base dispatcher for pairwise distance computation & reduction.

    Each dispatcher extending the base :class:`BaseDistancesReductionDispatcher`
    dispatcher must implement the :meth:`compute` classmethod.
    """

    @classmethod
    def valid_metrics(cls) -> List[str]:
        excluded = {'pyfunc', 'mahalanobis', 'hamming', *BOOL_METRICS}
        return sorted(({'sqeuclidean'} | set(METRIC_MAPPING64.keys())) - excluded)

    @classmethod
    def is_usable_for(cls, X, Y, metric) -> bool:
        """Return True if the dispatcher can be used for the
        given parameters.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples_X, n_features)
            Input data.

        Y : {ndarray, sparse matrix} of shape (n_samples_Y, n_features)
            Input data.

        metric : str, default='euclidean'
            The distance metric to use.
            For a list of available metrics, see the documentation of
            :class:`~sklearn.metrics.DistanceMetric`.

        Returns
        -------
        True if the dispatcher can be used, else False.
        """
        if issparse(X) and issparse(Y) and isinstance(metric, str) and ('euclidean' in metric):
            return False

        def is_numpy_c_ordered(X):
            return hasattr(X, 'flags') and getattr(X.flags, 'c_contiguous', False)

        def is_valid_sparse_matrix(X):
            return issparse(X) and X.format == 'csr' and (X.nnz > 0) and (X.indices.dtype == X.indptr.dtype == np.int32)
        is_usable = get_config().get('enable_cython_pairwise_dist', True) and (is_numpy_c_ordered(X) or is_valid_sparse_matrix(X)) and (is_numpy_c_ordered(Y) or is_valid_sparse_matrix(Y)) and (X.dtype == Y.dtype) and (X.dtype in (np.float32, np.float64)) and (metric in cls.valid_metrics() or isinstance(metric, DistanceMetric))
        return is_usable

    @classmethod
    @abstractmethod
    def compute(cls, X, Y, **kwargs):
        """Compute the reduction.

        Parameters
        ----------
        X : ndarray or CSR matrix of shape (n_samples_X, n_features)
            Input data.

        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)
            Input data.

        **kwargs : additional parameters for the reduction

        Notes
        -----
        This method is an abstract class method: it has to be implemented
        for all subclasses.
        """