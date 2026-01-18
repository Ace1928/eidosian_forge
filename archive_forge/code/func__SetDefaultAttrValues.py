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
def _SetDefaultAttrValues(node_def, op_def):
    """Set any default attr values in `node_def` that aren't present."""
    assert node_def.op == op_def.name
    for attr_def in op_def.attr:
        key = attr_def.name
        if attr_def.HasField('default_value'):
            value = node_def.attr[key]
            if value is None or value.WhichOneof('value') is None:
                node_def.attr[key].CopyFrom(attr_def.default_value)