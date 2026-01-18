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
def _merge_tensor_signatures(self, signatures):
    """Returns a tensor that merges the given signatures.

    Args:
      signatures: A dictionary of the signature updates from signature name to
      a tensor of dimension [1].
    Returns:
      A tensor that concats the signature values in a predefined order.
    Raises:
      ValueError: Unable to merge signatures.
    """
    sorted_update = []
    if self._num_signature_dimensions() > 1:
        signature_indices = self._signature_types()
        for _, val in sorted(signatures.items(), key=lambda item: signature_indices[item[0]]):
            sorted_update.append(val)
        updates = array_ops_stack.stack(sorted_update, axis=0, name='merge_single_op_signatures')
    elif self._num_signature_dimensions() == 1:
        (_, val), = signatures.items()
        updates = val
    else:
        raise ValueError('Cannot merge 0 signatures. Check the value passed for flag --signatures.')
    return updates