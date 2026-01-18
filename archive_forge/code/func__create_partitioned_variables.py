import collections
import copy
import math
import re
from typing import Optional
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util.tf_export import tf_export
def _create_partitioned_variables(name, num_hosts, vocabulary_size, embedding_dimension, initializer, collections=None):
    """Creates PartitionedVariables based on `num_hosts` for `table`."""
    num_slices = min(vocabulary_size, num_hosts)
    var_list = list(variable_scope.get_variable(name, shape=(vocabulary_size, embedding_dimension), partitioner=partitioned_variables.fixed_size_partitioner(num_slices), dtype=dtypes.float32, initializer=initializer, collections=collections, trainable=False))
    if vocabulary_size >= num_hosts:
        return var_list
    for idx in range(num_hosts - vocabulary_size):
        var_list.append(variable_scope.get_variable('dummy_{}_{}'.format(vocabulary_size + idx, name), shape=(1, embedding_dimension), dtype=dtypes.float32, initializer=initializer, collections=[ops.GraphKeys.LOCAL_VARIABLES], trainable=False))
    return var_list