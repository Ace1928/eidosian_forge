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