import json
import os
import sys
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_kernel_registry
def _get_ops_from_nodedefs(node_defs):
    """Gets the ops and kernels needed from the list of NodeDef.

  If a NodeDef's op is not in the allowlist of ops without kernel and there is
  no kernel found for this NodeDef, then skip that NodeDef and proceed to the
  next one.

  Args:
    node_defs: list of NodeDef's to get op/kernel information.

  Returns:
    A set of (op_name, kernel_name) tuples.
  """
    ops = set()
    for node_def in node_defs:
        op_and_kernel = get_ops_from_nodedef(node_def)
        if op_and_kernel:
            ops.add(op_and_kernel)
    return ops