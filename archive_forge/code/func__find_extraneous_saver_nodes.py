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
def _find_extraneous_saver_nodes(graph_def, saver_def):
    """Identifies any nodes in the graph_def related to unused Savers.

  This approach assumes that each Saver is cleanly isolated in its own name
  scope, so we need only identify the scopes associated with extraneous Savers
  and return all the nodes in those scopes.

  Args:
    graph_def: a GraphDef proto to evaluate.
    saver_def: a SaverDef proto referencing Save/Restore ops to be retained.
  Returns:
    An iterable of node names that may be safely omitted.
  """
    nodes = {node_def.name: (set((tensor.get_op_name(x) for x in node_def.input)), node_def.op) for node_def in graph_def.node}
    retain_scope_save = None
    retain_scope_restore = None
    if saver_def is not None:
        save_op_name = tensor.get_op_name(saver_def.save_tensor_name)
        restore_op_name = tensor.get_op_name(saver_def.restore_op_name)
        retain_scope_restore = _get_scope(restore_op_name) + '/'
        retain_scope_save = _get_scope(save_op_name) + '/'
    all_saver_node_names = set((name for name, (_, op) in nodes.items() if op in SAVE_AND_RESTORE_OPS))
    all_saver_scopes = set((_get_scope(x) for x in all_saver_node_names)) - all_saver_node_names
    all_saver_scopes = set((x + '/' for x in all_saver_scopes))
    extraneous_scopes = all_saver_scopes - set([retain_scope_save, retain_scope_restore])
    extraneous_node_names = set()
    for name, _ in nodes.items():
        for extraneous_scope in extraneous_scopes:
            if name.startswith(extraneous_scope):
                extraneous_node_names.add(name)
                break
    return extraneous_node_names