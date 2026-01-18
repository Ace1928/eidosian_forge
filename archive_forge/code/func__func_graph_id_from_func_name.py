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
def _func_graph_id_from_func_name(self, op_type):
    """Attempt to get the ID of a FuncGraph based on an op type name.

    Also caches the ID for faster access later.

    Args:
      op_type: Op type string, which may be the name of a function.

    Returns:
      If the op_type name does not fit the pattern of a function name (e.g.,
      one that starts with "__inference_"), `None` is returned immediately.
      Else, if the FuncGraph is found, ID of the underlying FuncGraph is
      returned as a string.
      Else, `None` is returned.
    """
    op_type = compat.as_bytes(op_type)
    if is_op_type_function(op_type):
        if op_type in self._op_type_to_context_id:
            return self._op_type_to_context_id[op_type]
        with self._context_lock:
            for function in self._function_to_graph_id:
                if function.name == op_type:
                    graph_id = self._function_to_graph_id[function]
                    self._op_type_to_context_id[op_type] = graph_id
                    return graph_id
        return None
    else:
        return None