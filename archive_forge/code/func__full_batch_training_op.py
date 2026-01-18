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
def _full_batch_training_op(self, inputs, num_clusters, cluster_idx_list, cluster_centers):
    """Creates an op for training for full batch case.

    Args:
      inputs: list of input Tensors.
      num_clusters: an integer Tensor providing the number of clusters.
      cluster_idx_list: A vector (or list of vectors). Each element in the
        vector corresponds to an input row in 'inp' and specifies the cluster id
        corresponding to the input.
      cluster_centers: Tensor Ref of cluster centers.

    Returns:
      An op for doing an update of mini-batch k-means.
    """
    cluster_sums = []
    cluster_counts = []
    epsilon = constant_op.constant(1e-06, dtype=inputs[0].dtype)
    for inp, cluster_idx in zip(inputs, cluster_idx_list):
        with ops.colocate_with(inp, ignore_existing=True):
            cluster_sums.append(math_ops.unsorted_segment_sum(inp, cluster_idx, num_clusters))
            cluster_counts.append(math_ops.unsorted_segment_sum(array_ops.reshape(array_ops.ones(array_ops.reshape(array_ops.shape(inp)[0], [-1])), [-1, 1]), cluster_idx, num_clusters))
    with ops.colocate_with(cluster_centers, ignore_existing=True):
        new_clusters_centers = math_ops.add_n(cluster_sums) / (math_ops.cast(math_ops.add_n(cluster_counts), cluster_sums[0].dtype) + epsilon)
        if self._clusters_l2_normalized():
            new_clusters_centers = nn_impl.l2_normalize(new_clusters_centers, dim=1)
    return state_ops.assign(cluster_centers, new_clusters_centers)