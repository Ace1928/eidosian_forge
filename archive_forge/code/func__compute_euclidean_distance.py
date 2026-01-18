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
@classmethod
def _compute_euclidean_distance(cls, inputs, clusters):
    """Computes Euclidean distance between each input and each cluster center.

    Args:
      inputs: list of input Tensors.
      clusters: cluster Tensor.

    Returns:
      list of Tensors, where each element corresponds to each element in inputs.
      The value is the distance of each row to all the cluster centers.
    """
    output = []
    for inp in inputs:
        with ops.colocate_with(inp, ignore_existing=True):
            squared_distance = math_ops.reduce_sum(math_ops.square(inp), 1, keepdims=True) - 2 * math_ops.matmul(inp, clusters, transpose_b=True) + array_ops.transpose(math_ops.reduce_sum(math_ops.square(clusters), 1, keepdims=True))
            output.append(squared_distance)
    return output