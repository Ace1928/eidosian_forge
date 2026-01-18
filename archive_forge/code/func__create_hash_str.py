import collections
import hashlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
def _create_hash_str(self, input_arg, output_arg, node_def):
    """Creates an 8-character string unique to this input.

    Args:
      input_arg: the input_arg field of an OpDef
                 (e.g. self._definition.signature.input_arg)
      output_arg: the output_arg field of an OpDef
                 (e.g. self._definition.signature.output_arg)
      node_def: the node_def field of a FunctionDef
                (e.g. self._definition.node_def)

    Returns:
      The unique string for this input
    """
    hasher = hashlib.sha1()

    def update_num(n):
        hasher.update(compat.as_bytes('%x' % n))

    def update_str(s):
        update_num(len(s))
        hasher.update(compat.as_bytes(s))

    def update_strs(slist):
        update_num(len(slist))
        for s in slist:
            update_str(s)
    for adef in input_arg:
        update_str(adef.SerializeToString())
    for adef in output_arg:
        update_str(adef.SerializeToString())
    for n in sorted(node_def, key=lambda n: n.name):
        update_str(n.name)
        update_str(n.op)
        update_strs(n.input)
        update_num(len(n.attr))
        for k in sorted(n.attr):
            update_str(k)
            update_str(n.attr[k].SerializeToString())
    return hasher.hexdigest()[:8]