import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _binary_assert_doc_v2(sym, opname, test_var):
    """Common docstring for v2 assert_* ops that compare two tensors element-wise.

  Args:
    sym: Binary operation symbol, i.e. "=="
    opname: Name for the symbol, i.e. "assert_equal"
    test_var: A number used in the docstring example

  Returns:
    Decorator that adds the appropriate docstring to the function for
  symbol `sym`.
  """

    def _decorator(func):
        """Decorator that adds docstring to the function for symbol `sym`.

    Args:
      func: Function for a TensorFlow op

    Returns:
      A version of `func` with documentation attached.
    """
        func.__doc__ = '\n    Assert the condition `x {sym} y` holds element-wise.\n\n    This Op checks that `x[i] {sym} y[i]` holds for every pair of (possibly\n    broadcast) elements of `x` and `y`. If both `x` and `y` are empty, this is\n    trivially satisfied.\n\n    If `x` {sym} `y` does not hold, `message`, as well as the first `summarize`\n    entries of `x` and `y` are printed, and `InvalidArgumentError` is raised.\n\n    When using inside `tf.function`, this API takes effects during execution.\n    It\'s recommended to use this API with `tf.control_dependencies` to\n    ensure the correct execution order.\n\n    In the following example, without `tf.control_dependencies`, errors may\n    not be raised at all.\n    Check `tf.control_dependencies` for more details.\n\n    >>> def check_size(x):\n    ...   with tf.control_dependencies([\n    ...       tf.debugging.{opname}(tf.size(x), {test_var},\n    ...                       message=\'Bad tensor size\')]):\n    ...     return x\n\n    >>> check_size(tf.ones([2, 3], tf.float32))\n    Traceback (most recent call last):\n       ...\n    InvalidArgumentError: ...\n\n    Args:\n      x:  Numeric `Tensor`.\n      y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.\n      message: A string to prefix to the default message. (optional)\n      summarize: Print this many entries of each tensor. (optional)\n      name: A name for this operation (optional).  Defaults to "{opname}".\n\n    Returns:\n      Op that raises `InvalidArgumentError` if `x {sym} y` is False. This can\n        be used with `tf.control_dependencies` inside of `tf.function`s to\n        block followup computation until the check has executed.\n      @compatibility(eager)\n      returns None\n      @end_compatibility\n\n    Raises:\n      InvalidArgumentError: if the check can be performed immediately and\n        `x == y` is False. The check can be performed immediately during eager\n        execution or if `x` and `y` are statically known.\n    '.format(sym=sym, opname=opname, test_var=test_var)
        return func
    return _decorator