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
def _mini_batch_training_op(self, inputs, cluster_idx_list, cluster_centers, total_counts):
    """Creates an op for training for mini batch case.

    Args:
      inputs: list of input Tensors.
      cluster_idx_list: A vector (or list of vectors). Each element in the
        vector corresponds to an input row in 'inp' and specifies the cluster id
        corresponding to the input.
      cluster_centers: Tensor Ref of cluster centers.
      total_counts: Tensor Ref of cluster counts.

    Returns:
      An op for doing an update of mini-batch k-means.
    """
    update_ops = []
    for inp, cluster_idx in zip(inputs, cluster_idx_list):
        with ops.colocate_with(inp, ignore_existing=True):
            assert total_counts is not None
            cluster_idx = array_ops.reshape(cluster_idx, [-1])
            unique_ids, unique_idx = array_ops.unique(cluster_idx)
            num_unique_cluster_idx = array_ops.size(unique_ids)
            with ops.colocate_with(total_counts, ignore_existing=True):
                old_counts = array_ops.gather(total_counts, unique_ids)
            with ops.colocate_with(cluster_centers, ignore_existing=True):
                old_cluster_centers = array_ops.gather(cluster_centers, unique_ids)
            count_updates = math_ops.unsorted_segment_sum(array_ops.ones_like(unique_idx, dtype=total_counts.dtype), unique_idx, num_unique_cluster_idx)
            cluster_center_updates = math_ops.unsorted_segment_sum(inp, unique_idx, num_unique_cluster_idx)
            broadcast_shape = array_ops.concat([array_ops.reshape(num_unique_cluster_idx, [1]), array_ops.ones(array_ops.reshape(array_ops.rank(inp) - 1, [1]), dtype=dtypes.int32)], 0)
            cluster_center_updates -= math_ops.cast(array_ops.reshape(count_updates, broadcast_shape), inp.dtype) * old_cluster_centers
            learning_rate = math_ops.reciprocal(math_ops.cast(old_counts + count_updates, inp.dtype))
            learning_rate = array_ops.reshape(learning_rate, broadcast_shape)
            cluster_center_updates *= learning_rate
        update_counts = state_ops.scatter_add(total_counts, unique_ids, count_updates)
        update_cluster_centers = state_ops.scatter_add(cluster_centers, unique_ids, cluster_center_updates)
        update_ops.extend([update_counts, update_cluster_centers])
    return control_flow_ops.group(*update_ops)