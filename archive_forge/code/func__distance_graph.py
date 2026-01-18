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
def _distance_graph(cls, inputs, clusters, distance_metric):
    """Computes distance between each input and each cluster center.

    Args:
      inputs: list of input Tensors.
      clusters: cluster Tensor.
      distance_metric: distance metric used for clustering

    Returns:
      list of Tensors, where each element corresponds to each element in inputs.
      The value is the distance of each row to all the cluster centers.
      Currently only Euclidean distance and cosine distance are supported.
    """
    assert isinstance(inputs, list)
    if distance_metric == SQUARED_EUCLIDEAN_DISTANCE:
        return cls._compute_euclidean_distance(inputs, clusters)
    elif distance_metric == COSINE_DISTANCE:
        return cls._compute_cosine_distance(inputs, clusters, inputs_normalized=True)
    else:
        assert False, str(distance_metric)