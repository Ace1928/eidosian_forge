import copy
from packaging import version as packaging_version  # pylint: disable=g-bad-import-order
import os.path
import re
import sys
from google.protobuf.any_pb2 import Any
from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def _should_include_node(node_or_node_name, export_scope, exclude_nodes):
    """Returns `True` if a node should be included.

  Args:
    node_or_node_name: A node or `string` node name.
    export_scope: `string`. Name scope under which to extract the subgraph. The
      scope name will be stripped from the node definitions for easy import
      later into new name scopes.
    exclude_nodes: An iterable of nodes or `string` node names to omit from the
      export, or None.  Note no sanity-checking is done, so this list must be
      carefully constructed to avoid producing an invalid graph.

  Returns:
    `True` if the node should be included.
  """
    if not isinstance(node_or_node_name, str):
        try:
            node_name = node_or_node_name.name
        except AttributeError:
            return True
    else:
        node_name = node_or_node_name
    if exclude_nodes and (node_or_node_name in exclude_nodes or node_name in exclude_nodes):
        return False
    return node_name.startswith(_UNBOUND_INPUT_PREFIX) or (not export_scope or node_name.startswith(export_scope))