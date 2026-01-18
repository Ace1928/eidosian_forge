import copy
import math
from typing import Sequence
import weakref
import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('distribute.experimental.partitioners.FixedShardsPartitioner', v1=[])
class FixedShardsPartitioner(Partitioner):
    """Partitioner that allocates a fixed number of shards.

  Examples:

  >>> # standalone usage:
  >>> partitioner = FixedShardsPartitioner(num_shards=2)
  >>> partitions = partitioner(tf.TensorShape([10, 3]), tf.float32)
  >>> [2, 1]
  >>>
  >>> # use in ParameterServerStrategy
  >>> # strategy = tf.distribute.experimental.ParameterServerStrategy(
  >>> #   cluster_resolver=cluster_resolver, variable_partitioner=partitioner)
  """

    def __init__(self, num_shards):
        """Creates a new `FixedShardsPartitioner`.

    Args:
      num_shards: `int`, number of shards to partition.
    """
        self._num_shards = num_shards

    def __call__(self, shape, dtype, axis=0):
        del dtype
        result = [1] * len(shape)
        result[axis] = min(self._num_shards, shape.dims[axis].value)
        return result