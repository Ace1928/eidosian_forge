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
@function.Defun(*_GetInputDtypes(func), func_name='%s_Wrapper' % func.name)
def Wrapper(*args):
    """A wrapper that handles loop-carried captured inputs."""
    result = func(*args)
    extra_args = tuple(function.get_extra_args())
    if isinstance(result, ops.Operation):
        return extra_args
    elif not isinstance(result, (list, tuple)):
        return (result,) + extra_args
    else:
        return result + type(result)(extra_args)