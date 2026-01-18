import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def _lookup_op_type(self, graph_id, op_name):
    """Lookup the type of an op by name and the immediately enclosing graph.

    Args:
      graph_id: Debugger-generated ID of the immediately-enclosing graph.
      op_name: Name of the op.

    Returns:
      Op type as a str.
    """
    return self._graph_by_id[graph_id].get_op_creation_digest(op_name).op_type