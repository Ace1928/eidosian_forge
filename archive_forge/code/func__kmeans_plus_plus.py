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
def _kmeans_plus_plus(self):
    inp = self._inputs[0]
    if self._distance_metric == COSINE_DISTANCE:
        inp = nn_impl.l2_normalize(inp, dim=1)
    return gen_clustering_ops.kmeans_plus_plus_initialization(inp, math_ops.cast(self._num_remaining, dtypes.int64), self._seed, self._kmeans_plus_plus_num_retries)