import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import variables
from tensorflow.python.util import nest
def _get_dtype_and_weakness(x):
    """Returns a TF type and weak type information from x.

  Args:
    x: an input scalar, array or a NumPy/TF/Python dtype.

  Raises:
    OverflowError: if Python int x is too large to convert to int32.
    NotImplementedError: when x is an unsupported input type.

  Returns:
    TF type and weak type information inferred from x in the form of
    (dtype, bool).
  """
    if isinstance(x, weak_tensor.WeakTensor):
        return (x.dtype, True)
    if isinstance(x, dtypes.DType):
        return (x, False)
    tf_dtype = getattr(x, 'dtype', None)
    if isinstance(tf_dtype, dtypes.DType):
        return (tf_dtype, False)
    if isinstance(x, (np.ndarray, np.generic)) or isinstance(tf_dtype, np.dtype):
        infer_dtype = dtypes.as_dtype(tf_dtype)
        return (infer_dtype, False)
    if isinstance(x, (bytes, str)) or tf_dtype in _all_str_dtypes:
        return _str
    try:
        if x in _NP_TO_TF:
            return (_NP_TO_TF[x], False)
    except TypeError:
        pass
    if isinstance(x, _pi):
        if x < np.iinfo(np.int32).min or x > np.iinfo(np.int32).max:
            raise OverflowError(f'Python int {x} too large to convert to np.int32')
        return _i32w
    if x == int:
        return _i32w
    if isinstance(x, _pf) or x == float:
        return _f32w
    if isinstance(x, _pc) or x == complex:
        return _c128w
    if isinstance(x, bool) or x == bool:
        return _b8
    if isinstance(x, tensor_shape.TensorShape):
        return _i32
    raise NotImplementedError(f'Auto dtype conversion semantics does not support {type(x)} type.')