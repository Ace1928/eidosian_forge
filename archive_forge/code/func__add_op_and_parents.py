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
def _add_op_and_parents(self, op):
    op_def = graph_to_function_def._get_op_def(op)
    if op._is_stateful and op not in self._allowlisted_stateful_ops:
        raise ValueError(f'Cannot capture a stateful node (name:{op.name}, type:{op.type}) by value.')
    elif op.type in ('Placeholder', 'PlaceholderV2'):
        raise ValueError(f'Cannot capture a placeholder (name:{op.name}, type:{op.type}) by value.')
    captured_inputs = [self._add_tensor_and_parents(x) for x in op.inputs]
    captured_op = self._create_op_internal(op.type, captured_inputs, [o.dtype for o in op.outputs], name=op.name, attrs=op.node_def.attr, op_def=op_def)
    for t, captured_t in zip(op.outputs, captured_op.outputs):
        self._captured[t.ref()] = captured_t
    return captured_op