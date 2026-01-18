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
def _reduce(tf_fn, a, axis=None, dtype=None, keepdims=None, promote_int=_TO_INT_, tf_bool_fn=None, preserve_bool=False):
    """A general reduction function.

  Args:
    tf_fn: the TF reduction function.
    a: the array to be reduced.
    axis: (optional) the axis along which to do the reduction. If None, all
      dimensions are reduced.
    dtype: (optional) the dtype of the result.
    keepdims: (optional) whether to keep the reduced dimension(s).
    promote_int: how to promote integer and bool inputs. There are three
      choices. (1) `_TO_INT_` always promotes them to np.int_ or np.uint; (2)
      `_TO_FLOAT` always promotes them to a float type (determined by
      dtypes.default_float_type); (3) None: don't promote.
    tf_bool_fn: (optional) the TF reduction function for bool inputs. It will
      only be used if `dtype` is explicitly set to `np.bool_` or if `a`'s dtype
      is `np.bool_` and `preserve_bool` is True.
    preserve_bool: a flag to control whether to use `tf_bool_fn` if `a`'s dtype
      is `np.bool_` (some reductions such as np.sum convert bools to integers,
      while others such as np.max preserve bools.

  Returns:
    An ndarray.
  """
    if dtype:
        dtype = np_utils.result_type(dtype)
    if keepdims is None:
        keepdims = False
    a = asarray(a, dtype=dtype)
    if (dtype == np.bool_ or (preserve_bool and a.dtype == np.bool_)) and tf_bool_fn is not None:
        return tf_bool_fn(input_tensor=a, axis=axis, keepdims=keepdims)
    if dtype is None:
        dtype = a.dtype.as_numpy_dtype
        if np.issubdtype(dtype, np.integer) or dtype == np.bool_:
            if promote_int == _TO_INT_:
                if dtype == np.bool_:
                    is_signed = True
                    width = 8
                else:
                    is_signed = np.issubdtype(dtype, np.signedinteger)
                    width = np.iinfo(dtype).bits
                if ops.is_auto_dtype_conversion_enabled():
                    if width < np.iinfo(np.int32).bits:
                        if is_signed:
                            dtype = np.int32
                        else:
                            dtype = np.uint32
                elif width < np.iinfo(np.int_).bits:
                    if is_signed:
                        dtype = np.int_
                    else:
                        dtype = np.uint
                a = math_ops.cast(a, dtype)
            elif promote_int == _TO_FLOAT:
                a = math_ops.cast(a, np_utils.result_type(float))
    if isinstance(axis, tensor_lib.Tensor) and axis.dtype not in (dtypes.int32, dtypes.int64):
        axis = math_ops.cast(axis, dtypes.int64)
    return tf_fn(input_tensor=a, axis=axis, keepdims=keepdims)