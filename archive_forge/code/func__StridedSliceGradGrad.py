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
@ops.RegisterGradient('StridedSliceGrad')
def _StridedSliceGradGrad(op, grad):
    """Gradient for StridedSliceGrad op."""
    begin = op.inputs[1]
    end = op.inputs[2]
    strides = op.inputs[3]
    return (None, None, None, None, array_ops.strided_slice(grad, begin, end, strides, begin_mask=op.get_attr('begin_mask'), end_mask=op.get_attr('end_mask'), ellipsis_mask=op.get_attr('ellipsis_mask'), new_axis_mask=op.get_attr('new_axis_mask'), shrink_axis_mask=op.get_attr('shrink_axis_mask')))