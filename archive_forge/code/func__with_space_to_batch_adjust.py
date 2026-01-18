import functools
import numbers
import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.platform import device_context
from tensorflow.python.platform import build_info
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
def _with_space_to_batch_adjust(orig, fill_value, spatial_dims):
    """Returns an `adjusted` version of `orig` based on `spatial_dims`.

  Tensor of the same type as `orig` and with shape
  `[max(spatial_dims), ...]` where:

    adjusted[spatial_dims[i] - 1, ...] = orig[i, ...]

  for 0 <= i < len(spatial_dims), and

    adjusted[j, ...] = fill_value

  for j != spatial_dims[i] - 1 for some i.

  If `orig` is a constant value, then the result will be a constant value.

  Args:
    orig: Tensor of rank > max(spatial_dims).
    fill_value: Numpy scalar (of same data type as `orig) specifying the fill
      value for non-spatial dimensions.
    spatial_dims: See with_space_to_batch.

  Returns:
    `adjusted` tensor.
  """
    fill_dims = orig.get_shape().as_list()[1:]
    dtype = orig.dtype.as_numpy_dtype
    parts = []
    const_orig = tensor_util.constant_value(orig)
    const_or_orig = const_orig if const_orig is not None else orig
    prev_spatial_dim = 0
    i = 0
    while i < len(spatial_dims):
        start_i = i
        start_spatial_dim = spatial_dims[i]
        if start_spatial_dim > 1:
            parts.append(np.full([start_spatial_dim - 1 - prev_spatial_dim] + fill_dims, fill_value, dtype=dtype))
        while i + 1 < len(spatial_dims) and spatial_dims[i + 1] == spatial_dims[i] + 1:
            i += 1
        parts.append(const_or_orig[start_i:i + 1])
        prev_spatial_dim = spatial_dims[i]
        i += 1
    if const_orig is not None:
        return np.concatenate(parts)
    else:
        return array_ops.concat(parts, 0)