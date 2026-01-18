from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as random_seed_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_clustering_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.embedding_ops import embedding_lookup
from tensorflow.python.ops.gen_clustering_ops import *
def _kmc2_multiple_centers(self):
    """Adds new initial cluster centers using the k-MC2 algorithm.

    In each call to the op, the provided batch is split into subsets based on
    the specified `kmc2_chain_length`. On each subset, a single Markov chain of
    the k-MC2 algorithm is used to add *one* new center cluster center. If there
    are less than `kmc2_chain_length` points in the subset, a single center is
    added using one Markov chain on the full input. It is assumed that the
    provided batch has previously been randomly permuted. Otherwise, k-MC2 may
    return suboptimal centers.

    Returns:
      An op that adds new cluster centers.
    """
    first_shard = self._inputs[0]
    batch_size = array_ops.shape(first_shard)[0]
    max_to_sample = math_ops.cast(batch_size / self._kmc2_chain_length, dtype=dtypes.int32)
    num_to_sample = math_ops.maximum(math_ops.minimum(self._num_remaining, max_to_sample), 1)

    def _cond(i, _):
        """Stopping condition for the while loop."""
        return math_ops.less(i, num_to_sample)

    def _body(i, _):
        """Body that adds a single new center based on a subset."""

        def _sample_random():
            """Returns a random point as a cluster center."""
            new_center = array_ops.reshape(first_shard[0], [1, -1])
            if self._distance_metric == COSINE_DISTANCE:
                new_center = nn_impl.l2_normalize(new_center, dim=1)
            return new_center

        def _sample_kmc2_chain():
            """Returns previous centers as well as a new center sampled using k-MC2."""
            start = i * self._kmc2_chain_length
            end = start + self._kmc2_chain_length
            subset = first_shard[start:end]
            _, distances = gen_clustering_ops.nearest_neighbors(subset, self._cluster_centers, 1)
            new_center_index = gen_clustering_ops.kmc2_chain_initialization(array_ops.squeeze(distances), self._seed)
            newly_sampled_center = array_ops.reshape(subset[new_center_index], [1, -1])
            if self._distance_metric == COSINE_DISTANCE:
                newly_sampled_center = nn_impl.l2_normalize(newly_sampled_center, dim=1)
            return array_ops.concat([self._cluster_centers, newly_sampled_center], 0)
        new_centers = cond.cond(math_ops.equal(self._num_selected, 0), _sample_random, _sample_kmc2_chain)
        assigned_centers = state_ops.assign(self._cluster_centers, new_centers, validate_shape=False)
        if self._cluster_centers_updated is not self._cluster_centers:
            assigned_centers = state_ops.assign(self._cluster_centers_updated, assigned_centers, validate_shape=False)
        return (i + 1, self._num_clusters - array_ops.shape(assigned_centers)[0])
    _, num_remaining = while_loop.while_loop(_cond, _body, [0, 0])
    return num_remaining