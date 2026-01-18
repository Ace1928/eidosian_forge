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
def _tensordot_axes(a, axes):
    """Generates two sets of contraction axes for the two tensor arguments."""
    a_shape = a.get_shape()
    if isinstance(axes, compat.integral_types):
        if axes < 0:
            raise ValueError(f'`axes` must be at least 0. Received: {axes}.')
        if a_shape.ndims is not None:
            if axes > a_shape.ndims:
                raise ValueError(f'`axes` must not be larger than the number of dimensions of tensor {a}.  Received {axes}, vs tensor dimensions {a_shape.ndims}.')
            return (list(builtins.range(a_shape.ndims - axes, a_shape.ndims)), list(builtins.range(axes)))
        else:
            rank = array_ops.rank(a)
            return (range(rank - axes, rank, dtype=dtypes.int32), range(axes, dtype=dtypes.int32))
    elif isinstance(axes, (list, tuple)):
        if len(axes) != 2:
            raise ValueError(f'`axes` must be an integer or have length 2. Received {axes}.')
        a_axes = axes[0]
        b_axes = axes[1]
        if isinstance(a_axes, compat.integral_types) and isinstance(b_axes, compat.integral_types):
            a_axes = [a_axes]
            b_axes = [b_axes]
        if len(a_axes) != len(b_axes):
            raise ValueError(f'Different number of contraction axes `a` and `b`, {len(a_axes)} != {len(b_axes)}.')
        return (a_axes, b_axes)
    else:
        axes = ops.convert_to_tensor(axes, name='axes', dtype=dtypes.int32)
        return (axes[0], axes[1])