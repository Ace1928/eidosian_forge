import builtins
import enum
import functools
import math
import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
def _with_index_update_helper(update_method, a, slice_spec, updates):
    """Implementation of ndarray._with_index_*."""
    if isinstance(slice_spec, bool) or (isinstance(slice_spec, tensor_lib.Tensor) and slice_spec.dtype == dtypes.bool) or (isinstance(slice_spec, (np.ndarray, np_arrays.ndarray)) and slice_spec.dtype == np.bool_):
        slice_spec = nonzero(slice_spec)
    if not isinstance(slice_spec, tuple):
        slice_spec = _as_spec_tuple(slice_spec)
    a_dtype = a.dtype
    a, updates = _promote_dtype_binary(a, updates)
    result_t = _slice_helper(a, slice_spec, update_method, updates)
    return result_t.astype(a_dtype)