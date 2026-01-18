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
def _get_strides_and_dilation_rate(num_spatial_dims, strides, dilation_rate):
    """Helper function for verifying strides and dilation_rate arguments.

  This is used by `convolution` and `pool`.

  Args:
    num_spatial_dims: int
    strides: Optional.  List of N ints >= 1.  Defaults to `[1]*N`.  If any value
      of strides is > 1, then all values of dilation_rate must be 1.
    dilation_rate: Optional.  List of N ints >= 1.  Defaults to `[1]*N`.  If any
      value of dilation_rate is > 1, then all values of strides must be 1.

  Returns:
    Normalized (strides, dilation_rate) as int32 numpy arrays of shape
    [num_spatial_dims].

  Raises:
    ValueError: if the parameters are invalid.
  """
    if dilation_rate is None:
        dilation_rate = [1] * num_spatial_dims
    elif len(dilation_rate) != num_spatial_dims:
        raise ValueError(f'`len(dilation_rate)` should be {num_spatial_dims}. Received: dilation_rate={dilation_rate} of length {len(dilation_rate)}')
    dilation_rate = np.array(dilation_rate, dtype=np.int32)
    if np.any(dilation_rate < 1):
        raise ValueError(f'all values of `dilation_rate` must be positive. Received: dilation_rate={dilation_rate}')
    if strides is None:
        strides = [1] * num_spatial_dims
    elif len(strides) != num_spatial_dims:
        raise ValueError(f'`len(strides)` should be {num_spatial_dims}. Received: strides={strides} of length {len(strides)}')
    strides = np.array(strides, dtype=np.int32)
    if np.any(strides < 1):
        raise ValueError(f'all values of `strides` must be positive. Received: strides={strides}')
    if np.any(strides > 1) and np.any(dilation_rate > 1):
        raise ValueError(f'`strides > 1` not supported in conjunction with `dilation_rate > 1`. Received: strides={strides} and dilation_rate={dilation_rate}')
    return (strides, dilation_rate)