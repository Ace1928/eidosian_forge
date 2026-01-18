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
def _validate_optimization_parameters(optimization_parameters, table_to_config_dict):
    """Validate global optimization_parameters and per table optimizers.

  If global optimizer is `None`, all table optimizers should be non `None`.

  Args:
      optimization_parameters: global optimizer provided in `TPUEmbedding`
        constructor.
      table_to_config_dict: A dictionary mapping from string of table name to
        `TableConfig`.
  """
    tbl_optimizer_missing = False
    for _, table_config in table_to_config_dict.items():
        if table_config.optimization_parameters is None:
            tbl_optimizer_missing = True
            break
    if optimization_parameters:
        if not isinstance(optimization_parameters, _OptimizationParameters):
            raise ValueError('`optimization_parameters` must inherit from `_OptimizationParameters`. `type(optimization_parameters)`={}'.format(type(optimization_parameters)))
    elif tbl_optimizer_missing:
        raise ValueError('`optimization_parameters` is missing.')