import collections
import functools
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def _slice_single_param(param, param_ndims_to_matrix_ndims, slices, batch_shape):
    """Slices into the batch shape of a single parameter.

  Args:
    param: The original parameter to slice; either a `Tensor` or an object
      with batch shape (LinearOperator).
    param_ndims_to_matrix_ndims: `int` number of right-most dimensions used for
      inferring matrix shape of the `LinearOperator`. For non-Tensor
      parameters, this is the number of this param's batch dimensions used by
      the matrix shape of the parent object.
    slices: iterable of slices received by `__getitem__`.
    batch_shape: The parameterized object's batch shape `Tensor`.

  Returns:
    new_param: Instance of the same type as `param`, batch-sliced according to
      `slices`.
  """
    param = _broadcast_parameter_with_batch_shape(param, param_ndims_to_matrix_ndims, array_ops.ones_like(batch_shape))
    if hasattr(param, 'batch_shape_tensor'):
        param_batch_shape = param.batch_shape_tensor()
    else:
        param_batch_shape = array_ops.shape(param)
    param_batch_rank = array_ops.size(param_batch_shape)
    param_batch_shape = param_batch_shape[:param_batch_rank - param_ndims_to_matrix_ndims]
    if tensor_util.constant_value(array_ops.size(batch_shape)) != 0 and tensor_util.constant_value(array_ops.size(param_batch_shape)) == 0:
        return param
    param_slices = _sanitize_slices(slices, intended_shape=batch_shape, deficient_shape=param_batch_shape)
    if param_ndims_to_matrix_ndims > 0:
        if Ellipsis not in [slc for slc in slices if not tensor_util.is_tensor(slc)]:
            param_slices.append(Ellipsis)
        param_slices += [slice(None)] * param_ndims_to_matrix_ndims
    return param.__getitem__(tuple(param_slices))