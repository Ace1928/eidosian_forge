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
def _make_tensor_trace_fun(self, tensor_name, tensor_trace_order):
    """Makes the tensor tracing function called by outside compilation.

    Args:
      tensor_name: name of the tensor being traced.
      tensor_trace_order: TensorTraceOrder object holding tensorname to id map.
    Returns:
      A function to be passed as the first argument to outside compilation.

    Raises:
      RuntimeError: If the trace mode is invalid.
    """

    def _print_tensor(tensor_name, num_elements, tensor, output_tensor):
        """Prints a tensor value to a file.

      Args:
        tensor_name: name of the tensor being traced.
        num_elements: number of elements to print (-1 means print all).
        tensor: the tensor needs to be returned.
        output_tensor: the tensor needs to be printed.

      Returns:
        The same tensor passed via the "tensor" argument.

      Raises:
        ValueError: If tensor_name is not already in
                    tensor_trace_order.tensorname_to_cache_idx.
      """
        if self._parameters.is_brief_mode():
            if tensor_name not in tensor_trace_order.tensorname_to_cache_idx:
                raise ValueError('Tensor %s with name %s is not in the tensorname_to_cache_idx' % (tensor, tensor_name))
            msg = '%d' % tensor_trace_order.tensorname_to_cache_idx[tensor_name]
        else:
            msg = '"%s"' % tensor_name
        if self._parameters.trace_dir:
            output_path = os.path.join(self._parameters.trace_dir, _TRACE_FILE_NAME + self._get_outfile_suffix())
            output_stream = _OUTPUT_STREAM_ESCAPE + output_path
        else:
            output_stream = sys.stderr
        return logging_ops.print_v2(msg, array_ops.shape(output_tensor), '@', self._replica_id, '\n', output_tensor, '\n', summarize=num_elements, output_stream=output_stream)

    def _show_part_tensor(tensor):
        """Trace function for printing part of the tensor."""
        return _print_tensor(tensor_name, _TRACE_MODE_PART_TENSOR_SIZE, tensor, tensor)

    def _show_full_tensor(tensor):
        """Trace function for printing the entire tensor."""
        return _print_tensor(tensor_name, -1, tensor, tensor)
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_PART_TENSOR:
        return _show_part_tensor
    if self._parameters.trace_mode in (tensor_tracer_flags.TRACE_MODE_NAN_INF, tensor_tracer_flags.TRACE_MODE_NORM, tensor_tracer_flags.TRACE_MODE_FULL_TENSOR, tensor_tracer_flags.TRACE_MODE_MAX_ABS, tensor_tracer_flags.TRACE_MODE_SUMMARY, tensor_tracer_flags.TRACE_MODE_HISTORY):
        return _show_full_tensor
    raise RuntimeError('Full tensor support is not available with trace mode %s' % self._parameters.trace_mode)