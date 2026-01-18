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
def _unary_assert_doc(sym, sym_name):
    """Common docstring for assert_* ops that evaluate a unary predicate over every element of a tensor.

  Args:
    sym: Mathematical symbol for the check performed on each element, i.e. "> 0"
    sym_name: English-language name for the op described by sym

  Returns:
    Decorator that adds the appropriate docstring to the function for symbol
    `sym`.
  """

    def _decorator(func):
        """Generated decorator that adds the appropriate docstring to the function for symbol `sym`.

    Args:
      func: Function for a TensorFlow op

    Returns:
      Version of `func` with documentation attached.
    """
        opname = func.__name__
        cap_sym_name = sym_name.capitalize()
        func.__doc__ = '\n    Assert the condition `x {sym}` holds element-wise.\n\n    When running in graph mode, you should add a dependency on this operation\n    to ensure that it runs. Example of adding a dependency to an operation:\n\n    ```python\n    with tf.control_dependencies([tf.debugging.{opname}(x, y)]):\n      output = tf.reduce_sum(x)\n    ```\n\n    {sym_name} means, for every element `x[i]` of `x`, we have `x[i] {sym}`.\n    If `x` is empty this is trivially satisfied.\n\n    Args:\n      x:  Numeric `Tensor`.\n      data:  The tensors to print out if the condition is False.  Defaults to\n        error message and first few entries of `x`.\n      summarize: Print this many entries of each tensor.\n      message: A string to prefix to the default message.\n      name: A name for this operation (optional).  Defaults to "{opname}".\n\n    Returns:\n      Op that raises `InvalidArgumentError` if `x {sym}` is False.\n      @compatibility(eager)\n        returns None\n      @end_compatibility\n\n    Raises:\n      InvalidArgumentError: if the check can be performed immediately and\n        `x {sym}` is False. The check can be performed immediately during\n        eager execution or if `x` is statically known.\n    '.format(sym=sym, sym_name=cap_sym_name, opname=opname)
        return func
    return _decorator