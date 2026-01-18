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
@ops.RegisterGradient('GatherNd')
def _GatherNdGrad(op, grad):
    ref = op.inputs[0]
    indices = op.inputs[1]
    ref_shape = array_ops.shape(ref, out_type=indices.dtype)
    if indices.shape.ndims == 2 and indices.shape.dims[-1].value == 1:
        ref_grad = indexed_slices_lib.IndexedSlices(grad, array_ops.squeeze(indices, axis=-1), ref_shape)
    else:
        ref_grad = array_ops.scatter_nd(indices, grad, ref_shape)
    return [ref_grad, None]