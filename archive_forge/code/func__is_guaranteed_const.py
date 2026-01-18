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
def _is_guaranteed_const(tensor):
    """Determines whether `tensor` is guaranteed to be a constant.

  A tensor is guaranteed to be a constant if either it was produced by
  a `GuaranteeConst` op or if all of its children are guaranteed to be
  constants.

  Args:
    tensor: The tensor for which to determine const-ness.

  Returns:
    True if `tensor` is guaranteed to be a constant, False otherwise.
  """
    if isinstance(tensor, ops.EagerTensor):
        return False

    class Work(object):

        def __init__(self, op, leaving):
            self.op = op
            self.leaving = leaving
    is_guaranteed_const = lambda op: op.node_def.op == 'GuaranteeConst'
    constants = set([])

    def all_inputs_const(op):
        return op.inputs and all((inp.op in constants for inp in op.inputs))
    visited = set([])
    stack = [Work(tensor.op, leaving=False)]
    while stack:
        work = stack.pop()
        if work.leaving:
            if all_inputs_const(work.op):
                constants.add(work.op)
            continue
        visited.add(work.op)
        if is_guaranteed_const(work.op):
            constants.add(work.op)
            continue
        stack.append(Work(work.op, leaving=True))
        for inp in work.op.inputs:
            if inp.op not in visited:
                stack.append(Work(inp.op, leaving=False))
    return tensor.op in constants