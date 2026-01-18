import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@def_function.function
def _load_variables_impl(config: Text, hosts: List[Tuple[int, Text]], variables: Dict[Text, Dict[Text, tf_variables.Variable]], table_config: tpu_embedding_v2_utils.TableConfig):
    """Load embedding tables to onto TPU for each table and host.

  Args:
    config: A serialized TPUEmbeddingConfiguration proto.
    hosts: A list of CPU devices, on per host.
    variables: A dictionary of dictionaries of TPUEmbeddingVariables. First key
      is the table name, second key is 'parameters' or the optimizer slot name.
    table_config: A list of tf.tpu.experimental.embedding.TableConfig objects.
  """

    def select_fn(host_id):

        def select_or_zeros(x):
            if host_id >= len(x.variables):
                return array_ops.zeros_like(x.variables[0])
            return x.variables[host_id]
        return select_or_zeros
    for host_id, host in enumerate(hosts):
        with ops.device(host):
            host_variables = nest.map_structure(select_fn(host_id), variables)
            for table in table_config:
                table.optimizer._load()(table_name=table.name, num_shards=len(hosts), shard_id=host_id, config=config, **host_variables[table.name])
                config = None