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
def _GetColocationNames(op):
    """Returns names of the ops that `op` should be colocated with."""
    colocation_names = []
    try:
        class_values = op.get_attr('_class')
    except ValueError:
        return
    for val in class_values:
        val = compat.as_str(val)
        if val.startswith('loc:@'):
            colocation_node_name = val[len('loc:@'):]
            if colocation_node_name != op.name:
                colocation_names.append(colocation_node_name)
    return colocation_names