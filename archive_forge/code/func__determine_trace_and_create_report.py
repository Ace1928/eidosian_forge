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
def _determine_trace_and_create_report(self, graph, ops_in_exec_path, graph_summary_tag):
    """Work needs to be done prior to TPU or CPU tracing.

    Args:
      graph: tf.graph
      ops_in_exec_path: Set of operations in the execution path.
      graph_summary_tag: the summary tag name for the given graph.
    Returns:
      An instance of tensor_tracer_report.TensorTraceOrder, containing list of
      tensors to be traced with their topological order information.
    Raises:
      RuntimeError: If opname filtering is incorrectly set.
    """
    self._check_trace_files()
    graph_order = tensor_tracer_report.sort_tensors_and_ops(graph)
    tensor_trace_points = graph.get_collection(_TENSOR_TRACER_COLLECTION)
    report_handler = tensor_tracer_report.TTReportHandle()
    traced_tensors = self._determine_and_instrument_traced_tensors(graph_order, ops_in_exec_path, tensor_trace_points, report_handler)
    logging.info('TensorTracer is tracing %d tensors.', len(traced_tensors))
    if traced_tensors and tensor_tracer_flags.TT_CHECK_FILTER.value:
        raise RuntimeError('Verify ops being traced by tensor tracer.')
    tensor_trace_order = tensor_tracer_report.TensorTraceOrder(graph_order, traced_tensors)
    num_signatures = self._num_signature_dimensions()
    if num_signatures and self._use_tensor_values_cache():
        if self._use_temp_cache():
            self._create_temp_cache(len(traced_tensors), num_signatures, graph)
        else:
            self._create_or_get_tensor_values_cache(_TT_SUMMARY_TAG, graph, [len(traced_tensors), num_signatures])
            if self._parameters.trace_mode in tensor_tracer_flags.TRACE_MODE_HISTORY:
                self._create_or_get_tensor_history_values_cache(_TT_SUMMARY_TAG, graph, [len(traced_tensors), num_signatures])
    if self._parameters.trace_mode in (tensor_tracer_flags.TRACE_MODE_SUMMARY, tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY):
        self._report_proto = report_handler.create_report_proto(self._tt_config, self._parameters, tensor_trace_order, tensor_trace_points, self._signature_types())
        if self._parameters.use_fingerprint_subdir:
            self._parameters.trace_dir = os.path.join(self._parameters.trace_dir, self._report_proto.fingerprint)
            logging.info('TensorTracer updating trace_dir to %s', self._parameters.trace_dir)
        self._report_proto_path = report_handler.report_proto_path(self._parameters.trace_dir, graph_summary_tag)
        if self._parameters.report_file_path != _SKIP_REPORT_FILE:
            report_handler.write_report_proto(self._report_proto_path, self._report_proto, self._parameters)
    elif self._parameters.trace_mode not in tensor_tracer_flags.TRACE_MODE_HISTORY:
        report_handler.create_report(self._tt_config, self._parameters, tensor_trace_order, tensor_trace_points)
    return tensor_trace_order