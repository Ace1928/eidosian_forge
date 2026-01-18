from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
@ops.RegisterGradient('MatrixDiagPartV3')
def _MatrixDiagPartV3Grad(op, grad):
    """Gradient for MatrixDiagPartV3."""
    matrix_shape = op.inputs[0].get_shape()[-2:]
    align = op.get_attr('align')
    if matrix_shape.is_fully_defined():
        return (array_ops.matrix_diag(grad, k=op.inputs[1], num_rows=matrix_shape[0], num_cols=matrix_shape[1], align=align), None, None)
    else:
        return (array_ops.matrix_set_diag(array_ops.zeros_like(op.inputs[0]), grad, k=op.inputs[1], align=align), None, None)