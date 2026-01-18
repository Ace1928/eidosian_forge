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
def _add_new_centers(self):
    """Adds some centers and returns the number of centers remaining."""
    new_centers = self._choose_initial_centers()
    if self._distance_metric == COSINE_DISTANCE:
        new_centers = nn_impl.l2_normalize(new_centers, dim=1)
    all_centers = cond.cond(math_ops.equal(self._num_selected, 0), lambda: new_centers, lambda: array_ops.concat([self._cluster_centers, new_centers], 0))
    a = state_ops.assign(self._cluster_centers, all_centers, validate_shape=False)
    if self._cluster_centers_updated is not self._cluster_centers:
        a = state_ops.assign(self._cluster_centers_updated, a, validate_shape=False)
    return self._num_clusters - array_ops.shape(a)[0]