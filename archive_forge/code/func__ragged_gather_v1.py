from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
@dispatch.dispatch_for_api(array_ops.gather)
def _ragged_gather_v1(params: ragged_tensor.RaggedOrDense, indices: ragged_tensor.RaggedOrDense, validate_indices=None, name=None, axis=0, batch_dims=0):
    return gather(params, indices, validate_indices, axis, batch_dims, name)