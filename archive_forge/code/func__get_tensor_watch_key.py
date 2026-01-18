import collections
import glob
import json
import os
import platform
import re
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def _get_tensor_watch_key(node_name, output_slot, debug_op):
    """Get the string representation of a debug watch on a tensor.

  Args:
    node_name: Name of the node by which the watched tensor is produced, as a
        string.
    output_slot: Output slot index of the tensor, as an integer.
    debug_op: Name of the debug op that is used to watch the tensor, as a
        string.

  Returns:
    A string representing the debug watch on the tensor (i.e., the "watch
        key").
  """
    return '%s:%s' % (_get_tensor_name(node_name, output_slot), debug_op)