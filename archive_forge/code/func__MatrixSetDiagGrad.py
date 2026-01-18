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
@ops.RegisterGradient('MatrixSetDiag')
def _MatrixSetDiagGrad(op, grad):
    """Gradient for MatrixSetDiag."""
    input_shape = op.inputs[0].get_shape().merge_with(grad.get_shape())
    diag_shape = op.inputs[1].get_shape()
    batch_shape = input_shape[:-2].merge_with(diag_shape[:-1])
    matrix_shape = input_shape[-2:]
    if batch_shape.is_fully_defined() and matrix_shape.is_fully_defined():
        diag_shape = batch_shape.as_list() + [min(matrix_shape.as_list())]
    else:
        with ops.colocate_with(grad):
            grad_shape = array_ops.shape(grad)
            grad_rank = array_ops.rank(grad)
            batch_shape = array_ops.slice(grad_shape, [0], [grad_rank - 2])
            matrix_shape = array_ops.slice(grad_shape, [grad_rank - 2], [2])
            min_dim = math_ops.reduce_min(matrix_shape)
            diag_shape = array_ops.concat([batch_shape, [min_dim]], 0)
    grad_input = array_ops.matrix_set_diag(grad, array_ops.zeros(diag_shape, dtype=grad.dtype))
    grad_diag = array_ops.matrix_diag_part(grad)
    return (grad_input, grad_diag)