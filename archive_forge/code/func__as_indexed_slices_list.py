import builtins
import numbers
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _as_indexed_slices_list(inputs, optimize=True):
    """Convert all elements of 'inputs' to IndexedSlices.

  Additionally, homogenize the types of all the indices to
  either int32 or int64.

  Args:
    inputs: List containing either Tensor or IndexedSlices objects.
    optimize: if true, attempt to optimize the conversion of each input.

  Returns:
    A list of IndexedSlices objects.

  Raises:
    TypeError: If 'inputs' is not a list or a tuple.
  """
    if not isinstance(inputs, (list, tuple)):
        raise TypeError(f'Expected a list or tuple, not {type(inputs)}.')
    outputs = [_as_indexed_slices(i, optimize=optimize) for i in inputs]
    with_int32_index = [o.indices for o in outputs if o.indices.dtype == dtypes.int32]
    if not with_int32_index or len(with_int32_index) == len(outputs):
        return outputs
    casted_outputs = []
    for o in outputs:
        if o.indices.dtype == dtypes.int32:
            casted_outputs.append(indexed_slices.IndexedSlices(o.values, cast(o.indices, dtypes.int64), o.dense_shape))
        else:
            casted_outputs.append(o)
    return casted_outputs