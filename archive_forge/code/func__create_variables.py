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
def _create_variables(self, num_clusters):
    """Creates variables.

    Args:
      num_clusters: an integer Tensor providing the number of clusters.

    Returns:
      Tuple with following elements:
      - cluster_centers: a Tensor for storing cluster centers
      - cluster_centers_initialized: bool Variable indicating whether clusters
            are initialized.
      - cluster_counts: a Tensor for storing counts of points assigned to this
            cluster. This is used by mini-batch training.
      - cluster_centers_updated: Tensor representing copy of cluster centers
            that are updated every step.
      - update_in_steps: numbers of steps left before we sync
            cluster_centers_updated back to cluster_centers.
    """
    init_value = array_ops.placeholder_with_default([], shape=None)
    cluster_centers = variable_v1.VariableV1(init_value, name=CLUSTERS_VAR_NAME, validate_shape=False)
    cluster_centers_initialized = variable_v1.VariableV1(False, dtype=dtypes.bool, name='initialized')
    if self._use_mini_batch and self._mini_batch_steps_per_iteration > 1:
        cluster_centers_updated = variable_v1.VariableV1(init_value, name='clusters_updated', validate_shape=False)
        update_in_steps = variable_v1.VariableV1(self._mini_batch_steps_per_iteration, dtype=dtypes.int64, name='update_in_steps')
        cluster_counts = variable_v1.VariableV1(array_ops.zeros([num_clusters], dtype=dtypes.int64))
    else:
        cluster_centers_updated = cluster_centers
        update_in_steps = None
        cluster_counts = variable_v1.VariableV1(array_ops.ones([num_clusters], dtype=dtypes.int64)) if self._use_mini_batch else None
    return (cluster_centers, cluster_centers_initialized, cluster_counts, cluster_centers_updated, update_in_steps)