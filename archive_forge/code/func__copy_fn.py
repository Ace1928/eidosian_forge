import abc
import contextlib
import types
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _copy_fn(fn):
    """Create a deep copy of fn.

  Args:
    fn: a callable

  Returns:
    A `FunctionType`: a deep copy of fn.

  Raises:
    TypeError: if `fn` is not a callable.
  """
    if not callable(fn):
        raise TypeError('fn is not callable: %s' % fn)
    return types.FunctionType(code=fn.__code__, globals=fn.__globals__, name=fn.__name__, argdefs=fn.__defaults__, closure=fn.__closure__)