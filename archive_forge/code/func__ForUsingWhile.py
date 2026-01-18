from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.gen_functional_ops import remote_call
from tensorflow.python.ops.gen_functional_ops import symbolic_gradient
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _ForUsingWhile(start, limit, delta, inputs, forbody, name=None, hostmem=None):
    """Helper to implement a For loop using a While."""
    d = math_ops.abs(delta)
    n = math_ops.cast(math_ops.cast(math_ops.abs(limit - start) + d - 1, dtypes.float32) / math_ops.cast(d, dtypes.float32), dtypes.int32)
    body_sig = [dtypes.int32] * 4 + list(forbody.declared_input_types)[1:]
    cond_name = '%s_Cond' % forbody.name

    @function.Defun(*body_sig, func_name=cond_name)
    def WhileCond(i, n, *args):
        del args
        return i < n
    body_name = '%s_Body' % forbody.name

    @function.Defun(*body_sig, func_name=body_name)
    def WhileBody(i, n, start, delta, *args):
        """A While wrapper for forbody that handles loop-carried captured inputs."""
        for_result = forbody(start + i * delta, *args)
        if isinstance(for_result, ops.Operation):
            for_result = ()
        elif isinstance(for_result, tensor.Tensor):
            for_result = (for_result,)
        return (i + 1, n, start, delta) + tuple(for_result)
    if hostmem is not None:
        hostmem = [0, 1, 2, 3] + [4 + _ for _ in hostmem]
    else:
        hostmem = [0, 1, 2, 3]
    results = While(input_=[0, n, start, delta] + inputs, cond=WhileCond, body=WhileBody, name=name, hostmem=hostmem)
    return list(results[4:len(results)])