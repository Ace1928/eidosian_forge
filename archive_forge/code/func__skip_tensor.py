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
def _skip_tensor(self, op_id, out_tensor, report_handler):
    """Returns True if we should not trace out_tensor.

    Args:
      op_id: Topological index of the op producing tensor.
      out_tensor: tf.Tensor
      report_handler: An instance of tensor_tracer_report.TTReportHandle.
    Returns:
      True if the tensor should not be traced, false otherwise.
    """
    non_numeric_tensor_types = set([dtypes.variant, dtypes.resource, dtypes.string])
    if out_tensor.dtype in non_numeric_tensor_types:
        report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_NON_NUMERIC_TENSOR))
        return True
    if [consumer for consumer in out_tensor.consumers() if TensorTracer.while_loop_op(consumer)]:
        report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_FEEDS_WHILELOOP_OP))
        return True
    if self._is_user_included_op(out_tensor.op):
        report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_USER_INCLUDED))
        if tensor_tracer_flags.TT_CHECK_FILTER.value:
            logging.info('USER_INCLUDED tensor %s', out_tensor.name)
        return False
    if self._is_user_excluded_op(out_tensor.op):
        report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_USER_EXCLUDED))
        if tensor_tracer_flags.TT_CHECK_FILTER.value:
            logging.info('USER_EXCLUDED tensor %s', out_tensor.name)
        return True
    if not out_tensor.get_shape().is_fully_defined():
        if self._parameters.trace_mode in (tensor_tracer_flags.TRACE_MODE_NAN_INF, tensor_tracer_flags.TRACE_MODE_NORM, tensor_tracer_flags.TRACE_MODE_HISTORY, tensor_tracer_flags.TRACE_MODE_MAX_ABS, tensor_tracer_flags.TRACE_MODE_SUMMARY):
            report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_TENSOR_GET_TRACED))
            return False
        else:
            report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_DYNAMIC_SHAPE))
            return True
    rank = len(out_tensor.shape)
    if rank < 1:
        if self._parameters.trace_scalar_ops:
            if TensorTracer.unsafe_scalar_trace(out_tensor.op):
                report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_UNSAFE_SCALAR))
                return True
            else:
                report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_SCALAR_GET_TRACED))
                return False
        else:
            report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_SKIP_SCALAR))
            return True
    else:
        report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_TENSOR_GET_TRACED))
        return False