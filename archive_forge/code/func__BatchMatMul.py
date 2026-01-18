import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
@ops.RegisterGradient('BatchMatMul')
def _BatchMatMul(op, grad):
    """Returns the gradient of x and y given the gradient of x * y."""
    x = op.inputs[0]
    y = op.inputs[1]
    adj_x = op.get_attr('adj_x')
    adj_y = op.get_attr('adj_y')
    if not adj_x:
        if not adj_y:
            grad_x = math_ops.matmul(grad, y, adjoint_a=False, adjoint_b=True, grad_a=True)
            grad_y = math_ops.matmul(x, grad, adjoint_a=True, adjoint_b=False, grad_b=True)
        else:
            grad_x = math_ops.matmul(grad, y, adjoint_a=False, adjoint_b=False, grad_a=True)
            grad_y = math_ops.matmul(grad, x, adjoint_a=True, adjoint_b=False, grad_b=True)
    elif not adj_y:
        grad_x = math_ops.matmul(y, grad, adjoint_a=False, adjoint_b=True, grad_a=True)
        grad_y = math_ops.matmul(x, grad, adjoint_a=False, adjoint_b=False, grad_b=True)
    else:
        grad_x = math_ops.matmul(y, grad, adjoint_a=True, adjoint_b=True, grad_a=True)
        grad_y = math_ops.matmul(grad, x, adjoint_a=True, adjoint_b=True, grad_b=True)
    return (grad_x, grad_y)