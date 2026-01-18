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
class ArgKminClassMode(BaseDistancesReductionDispatcher):
    """Compute the argkmin of row vectors of X on the ones of Y with labels.

    For each row vector of X, computes the indices of k first the rows
    vectors of Y with the smallest distances. Computes weighted mode of labels.

    ArgKminClassMode is typically used to perform bruteforce k-nearest neighbors
    queries when the weighted mode of the labels for the k-nearest neighbors
    are required, such as in `predict` methods.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def valid_metrics(cls) -> List[str]:
        excluded = {'euclidean', 'sqeuclidean'}
        return list(set(BaseDistancesReductionDispatcher.valid_metrics()) - excluded)

    @classmethod
    def compute(cls, X, Y, k, weights, Y_labels, unique_Y_labels, metric='euclidean', chunk_size=None, metric_kwargs=None, strategy=None):
        """Compute the argkmin reduction.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            The input array to be labelled.

        Y : ndarray of shape (n_samples_Y, n_features)
            The input array whose class membership are provided through the
            `Y_labels` parameter.

        k : int
            The number of nearest neighbors to consider.

        weights : ndarray
            The weights applied over the `Y_labels` of `Y` when computing the
            weighted mode of the labels.

        Y_labels : ndarray
            An array containing the index of the class membership of the
            associated samples in `Y`. This is used in labeling `X`.

        unique_Y_labels : ndarray
            An array containing all unique indices contained in the
            corresponding `Y_labels` array.

        metric : str, default='euclidean'
            The distance metric to use. For a list of available metrics, see
            the documentation of :class:`~sklearn.metrics.DistanceMetric`.
            Currently does not support `'precomputed'`.

        chunk_size : int, default=None,
            The number of vectors per chunk. If None (default) looks-up in
            scikit-learn configuration for `pairwise_dist_chunk_size`,
            and use 256 if it is not set.

        metric_kwargs : dict, default=None
            Keyword arguments to pass to specified metric function.

        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None
            The chunking strategy defining which dataset parallelization are made on.

            For both strategies the computations happens with two nested loops,
            respectively on chunks of X and chunks of Y.
            Strategies differs on which loop (outer or inner) is made to run
            in parallel with the Cython `prange` construct:

              - 'parallel_on_X' dispatches chunks of X uniformly on threads.
                Each thread then iterates on all the chunks of Y. This strategy is
                embarrassingly parallel and comes with no datastructures
                synchronisation.

              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.
                Each thread processes all the chunks of X in turn. This strategy is
                a sequence of embarrassingly parallel subtasks (the inner loop on Y
                chunks) with intermediate datastructures synchronisation at each
                iteration of the sequential outer loop on X chunks.

              - 'auto' relies on a simple heuristic to choose between
                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,
                'parallel_on_X' is usually the most efficient strategy.
                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'
                brings more opportunity for parallelism and is therefore more efficient
                despite the synchronization step at each iteration of the outer loop
                on chunks of `X`.

              - None (default) looks-up in scikit-learn configuration for
                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.

        Returns
        -------
        probabilities : ndarray of shape (n_samples_X, n_classes)
            An array containing the class probabilities for each sample.

        Notes
        -----
        This classmethod is responsible for introspecting the arguments
        values to dispatch to the most appropriate implementation of
        :class:`PairwiseDistancesArgKmin`.

        This allows decoupling the API entirely from the implementation details
        whilst maintaining RAII: all temporarily allocated datastructures necessary
        for the concrete implementation are therefore freed when this classmethod
        returns.
        """
        if weights not in {'uniform', 'distance'}:
            raise ValueError(f"Only the 'uniform' or 'distance' weights options are supported at this time. Got: weights={weights!r}.")
        if X.dtype == Y.dtype == np.float64:
            return ArgKminClassMode64.compute(X=X, Y=Y, k=k, weights=weights, Y_labels=np.array(Y_labels, dtype=np.intp), unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp), metric=metric, chunk_size=chunk_size, metric_kwargs=metric_kwargs, strategy=strategy)
        if X.dtype == Y.dtype == np.float32:
            return ArgKminClassMode32.compute(X=X, Y=Y, k=k, weights=weights, Y_labels=np.array(Y_labels, dtype=np.intp), unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp), metric=metric, chunk_size=chunk_size, metric_kwargs=metric_kwargs, strategy=strategy)
        raise ValueError(f'Only float64 or float32 datasets pairs are supported at this time, got: X.dtype={X.dtype} and Y.dtype={Y.dtype}.')