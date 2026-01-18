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
def _save_tensor_value_to_tmp_cache(self, cache_idx, updates, graph):
    """Returns an op that will save the given updates to an entry in the cache.

    Args:
      cache_idx: The cache index of the tensor within the cache.
      updates: A dictionary of the signature updates from signature name to
      a tensor of dimension [1].
      graph: A TensorFlow graph.
    Raises:
      RuntimeError:
        (1) graph is not already in self._temp_cache_var, or
        (2) cache_idx is out of range.
    """
    updates = self._merge_tensor_signatures(updates)
    updates = array_ops.reshape(updates, [self._num_signature_dimensions()])
    if graph not in self._temp_cache_var:
        raise RuntimeError('graph is not in self._temp_cache_var')
    if cache_idx >= len(self._temp_cache_var[graph]):
        raise RuntimeError('cache_idx (%d) is out of range (%d)' % (cache_idx, len(self._temp_cache_var[graph])))
    self._temp_cache_var[graph][cache_idx] = updates