import contextlib
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_util
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _RemoveDefaultAttrs(producer_op_list, graph_def):
    """Removes unknown default attrs according to `producer_op_list`.

  Removes any unknown attrs in `graph_def` (i.e. attrs that do not appear in
  registered OpDefs) that have a default value in `producer_op_list`.

  Args:
    producer_op_list: OpList proto.
    graph_def: GraphDef proto
  """
    producer_op_dict = {op.name: op for op in producer_op_list.op}
    for node in graph_def.node:
        if node.op in producer_op_dict:
            op_def = op_def_registry.get(node.op)
            if op_def is None:
                continue
            producer_op_def = producer_op_dict[node.op]
            for key in list(node.attr):
                if _FindAttrInOpDef(key, op_def) is None:
                    attr_def = _FindAttrInOpDef(key, producer_op_def)
                    if attr_def and attr_def.HasField('default_value') and (node.attr[key] == attr_def.default_value):
                        del node.attr[key]