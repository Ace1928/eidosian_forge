import abc
import os
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training import session_manager as sm
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import function_utils
from tensorflow.python.util.tf_export import tf_export
def _merge_run_options(self, options, incoming_options):
    """Merge two instances of RunOptions into the first one.

    During the merger, the numerical fields including trace_level,
    timeout_in_ms, inter_op_thread_pool are set to the larger one of the two.
    The boolean value is set to the logical OR of the two.
    debug_tensor_watch_opts of the original options is extended with that from
    the incoming one.

    Args:
      options: The options to merge into.
      incoming_options: The options to be merged into the first argument.
    """
    options.trace_level = max(options.trace_level, incoming_options.trace_level)
    options.timeout_in_ms = max(options.timeout_in_ms, incoming_options.timeout_in_ms)
    options.inter_op_thread_pool = max(options.inter_op_thread_pool, incoming_options.inter_op_thread_pool)
    options.output_partition_graphs = max(options.output_partition_graphs, incoming_options.output_partition_graphs)
    options.debug_options.debug_tensor_watch_opts.extend(incoming_options.debug_options.debug_tensor_watch_opts)
    options.debug_options.reset_disk_byte_usage = options.debug_options.reset_disk_byte_usage or incoming_options.debug_options.reset_disk_byte_usage
    options.report_tensor_allocations_upon_oom = options.report_tensor_allocations_upon_oom or incoming_options.report_tensor_allocations_upon_oom