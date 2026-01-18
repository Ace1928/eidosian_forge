from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
@ops.RegisterGradient('MatrixInverse')
def _MatrixInverseGrad(op, grad):
    """Gradient for MatrixInverse."""
    ainv = op.outputs[0]
    op_adjoint = op.get_attr('adjoint')
    return -math_ops.matmul(ainv, math_ops.matmul(grad, ainv, adjoint_a=op_adjoint, adjoint_b=not op_adjoint), adjoint_a=not op_adjoint)