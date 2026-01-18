import atexit
import os
import re
import socket
import threading
import uuid
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import function as function_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
def _get_outer_context_id(self, graph):
    """Get the ID of the immediate outer context of the input graph.

    Args:
      graph: The graph (context) in question.

    Returns:
      If an outer context exists, the immediate outer context name as a string.
      If such as outer context does not exist (i.e., `graph` is itself
      outermost), `None`.
    """
    if hasattr(graph, 'outer_graph') and graph.outer_graph:
        return self._get_context_id(graph.outer_graph)
    else:
        return None