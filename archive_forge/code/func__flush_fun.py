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
def _flush_fun(cache, replica_id, step_num):
    """Flushes the cache to a file corresponding to replica_id."""

    def _f(file_index):
        """Generates a func that flushes the cache to a file."""

        def _print_cache():
            """Flushes the cache to a file."""
            replica_str = '%d' % file_index
            if self._parameters.trace_dir:
                output_path = os.path.join(self._parameters.trace_dir, _COMPACT_TRACE_FILE_PREFIX) + replica_str + self._get_outfile_suffix()
                output_stream = _OUTPUT_STREAM_ESCAPE + output_path
            else:
                output_stream = sys.stderr
            new_step_line = _REPLICA_ID_TAG + replica_str
            print_ops = []
            if self._parameters.inspect_trace:
                if self._num_signature_dimensions() > 1:
                    raise ValueError('Inspecting multi signatures are not supported.')
                if self._parameters.trace_mode in tensor_tracer_flags.TRACE_MODE_HISTORY:
                    print_ops.append(self._inspect_history_cache(cache=cache, replica_id=replica_id, step_num=step_num, tensor_trace_order=tensor_trace_order))
                else:
                    print_ops.append(self._inspect_summary_cache(cache=cache, replica_id=replica_id, step_num=step_num, output_stream=output_stream, tensor_trace_order=tensor_trace_order))
            else:
                for i in range(self._num_signature_dimensions()):
                    print_ops.append(logging_ops.print_v2(new_step_line, '\n', cache[:, i], '\n', summarize=-1, output_stream=output_stream))
            with ops.control_dependencies(print_ops):
                return constant_op.constant(0).op
        return _print_cache

    def _eq(file_index):
        return math_ops.equal(replica_id, file_index)
    flush_op_cases = {}
    flush_op_cases[_eq(0)] = _f(0)
    for i in range(1, num_replicas):
        if on_tpu and (not self._parameters.collect_summary_per_core):
            flush_op_cases[_eq(i)] = control_flow_ops.no_op
        else:
            flush_op_cases[_eq(i)] = _f(i)
    return control_flow_case.case(flush_op_cases, exclusive=True)