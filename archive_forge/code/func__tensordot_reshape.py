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
def _tensordot_reshape(a, axes, flipped=False):
    """Helper method to perform transpose and reshape for contraction op.

    This method is helpful in reducing `math_ops.tensordot` to `math_ops.matmul`
    using `array_ops.transpose` and `array_ops.reshape`. The method takes a
    tensor and performs the correct transpose and reshape operation for a given
    set of indices. It returns the reshaped tensor as well as a list of indices
    necessary to reshape the tensor again after matrix multiplication.

    Args:
      a: `Tensor`.
      axes: List or `int32` `Tensor` of unique indices specifying valid axes of
        `a`.
      flipped: An optional `bool`. Defaults to `False`. If `True`, the method
        assumes that `a` is the second argument in the contraction operation.

    Returns:
      A tuple `(reshaped_a, free_dims, free_dims_static)` where `reshaped_a` is
      the tensor `a` reshaped to allow contraction via `matmul`, `free_dims` is
      either a list of integers or an `int32` `Tensor`, depending on whether
      the shape of a is fully specified, and free_dims_static is either a list
      of integers and None values, or None, representing the inferred
      static shape of the free dimensions
    """
    if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
        shape_a = a.get_shape().as_list()
        axes = [i if i >= 0 else i + len(shape_a) for i in axes]
        free = [i for i in builtins.range(len(shape_a)) if i not in axes]
        free_dims = [shape_a[i] for i in free]
        prod_free = int(np.prod([shape_a[i] for i in free]))
        prod_axes = int(np.prod([shape_a[i] for i in axes]))
        perm = list(axes) + free if flipped else free + list(axes)
        new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
        if (perm != np.arange(len(shape_a))).any():
            a_trans = array_ops.transpose(a, perm)
        else:
            a_trans = a
        if a_trans.get_shape().as_list() != new_shape:
            reshaped_a = array_ops.reshape(a_trans, new_shape)
        else:
            reshaped_a = a_trans
        return (reshaped_a, free_dims, free_dims)
    else:
        if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in builtins.range(len(shape_a)) if i not in axes]
            axes_dims = [shape_a[i] for i in axes]
            free_dims = [shape_a[i] for i in free]
            free_dims_static = free_dims
            axes = ops.convert_to_tensor(axes, dtype=dtypes.int32, name='axes')
            free = ops.convert_to_tensor(free, dtype=dtypes.int32, name='free')
            shape_a = array_ops.shape(a)
        else:
            free_dims_static = None
            shape_a = array_ops.shape(a)
            rank_a = array_ops.rank(a)
            axes = ops.convert_to_tensor(axes, dtype=dtypes.int32, name='axes')
            axes = array_ops.where(axes >= 0, axes, axes + rank_a)
            free, _ = gen_array_ops.list_diff(range(rank_a), axes, dtypes.int32)
        free_dims = array_ops.gather(shape_a, free)
        axes_dims = array_ops.gather(shape_a, axes)
        prod_free_dims = reduce_prod(free_dims)
        prod_axes_dims = reduce_prod(axes_dims)
        if flipped:
            perm = array_ops.concat([axes, free], 0)
            new_shape = array_ops_stack.stack([prod_axes_dims, prod_free_dims])
        else:
            perm = array_ops.concat([free, axes], 0)
            new_shape = array_ops_stack.stack([prod_free_dims, prod_axes_dims])
        reshaped_a = array_ops.reshape(array_ops.transpose(a, perm), new_shape)
        return (reshaped_a, free_dims, free_dims_static)