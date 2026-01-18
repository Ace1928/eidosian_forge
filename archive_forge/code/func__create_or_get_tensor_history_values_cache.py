import collections
import hashlib
import operator
import os
import os.path
import sys
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import analytics
from tensorflow.python.platform import gfile
from tensorflow.python.platform import remote_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.tpu import tensor_tracer_flags
from tensorflow.python.tpu import tensor_tracer_report
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import training_util
def _create_or_get_tensor_history_values_cache(self, cache_name, graph, shape=None, dtype=dtypes.float32):
    """Creates a variable as the cache to store historic intermediate tensor values.

    Args:
      cache_name: Name to be given to the cache (an instance of tf.variable).
      graph: Tensorflow graph.
      shape: A list of dimensions.
      dtype: Data type of created cache.
    Returns:
      A ref to newly created or existing cache with the given dimensions.
    Raises:
      ValueError:
        (1) If graph is None, or
        (2) shape is None when a new cache needs to be created.
    """
    if graph is None:
        raise ValueError('Invalid graph.')
    if graph not in self._history_value_cache:
        self._history_value_cache[graph] = {}
    if cache_name not in self._history_value_cache[graph]:
        if shape is None:
            raise ValueError('shape must be provided at cache creation.')
        if dtype.is_integer:
            init_val = int(_COMPACT_TRACE_ENTRY_INIT_VALUE)
        else:
            init_val = _COMPACT_TRACE_ENTRY_INIT_VALUE
        with graph.as_default() as g, g.name_scope(None):
            self._history_value_cache[graph][cache_name] = variable_scope.get_variable('tt_history' + '_' + self._escape_namescopes(cache_name), shape=shape, dtype=dtype, initializer=init_ops.constant_initializer(init_val), trainable=False, use_resource=True, collections=[_TENSOR_TRACER_STORAGE, ops.GraphKeys.LOCAL_VARIABLES])
    return self._history_value_cache[graph][cache_name]