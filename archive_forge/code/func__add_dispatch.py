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
@tf_export('__operators__.add', v1=[])
@dispatch.add_dispatch_support
def _add_dispatch(x, y, name=None):
    """The operation invoked by the `Tensor.__add__` operator.

  Purpose in the API:

    This method is exposed in TensorFlow's API so that library developers
    can register dispatching for `Tensor.__add__` to allow it to handle
    custom composite tensors & other custom objects.

    The API symbol is not intended to be called by users directly and does
    appear in TensorFlow's generated documentation.

  Args:
    x: The left-hand side of the `+` operator.
    y: The right-hand side of the `+` operator.
    name: an optional name for the operation.

  Returns:
    The result of the elementwise `+` operation.
  """
    if not ops.is_auto_dtype_conversion_enabled() and (not isinstance(y, tensor_lib.Tensor)) and (not isinstance(y, sparse_tensor.SparseTensor)):
        y = ops.convert_to_tensor(y, dtype_hint=x.dtype.base_dtype, name='y')
    return add(x, y, name=name)