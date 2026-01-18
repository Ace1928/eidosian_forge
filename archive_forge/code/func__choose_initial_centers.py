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
def _choose_initial_centers(self):
    if isinstance(self._initial_clusters, str):
        if self._initial_clusters == RANDOM_INIT:
            return self._greedy_batch_sampler(self._random)
        else:
            return self._single_batch_sampler(self._kmeans_plus_plus)
    elif callable(self._initial_clusters):
        return self._initial_clusters(self._inputs, self._num_remaining)
    else:
        with ops.control_dependencies([check_ops.assert_equal(self._num_remaining, array_ops.shape(self._initial_clusters)[0])]):
            return self._initial_clusters