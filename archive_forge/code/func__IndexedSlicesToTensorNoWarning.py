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
def _IndexedSlicesToTensorNoWarning(indexed_slices):
    """Converts an IndexedSlices to a Tensor without sparse->dense warnings."""
    if not isinstance(indexed_slices, indexed_slices_lib.IndexedSlices):
        return indexed_slices
    if indexed_slices.dense_shape is None:
        raise ValueError('Tensor conversion requested for IndexedSlices without dense_shape: %s' % str(indexed_slices))
    return math_ops.unsorted_segment_sum(indexed_slices.values, indexed_slices.indices, indexed_slices.dense_shape[0])