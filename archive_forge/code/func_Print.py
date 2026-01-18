import collections as py_collections
import os
import pprint
import random
import sys
from absl import logging
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.gen_logging_ops import *
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
@deprecated('2018-08-20', 'Use tf.print instead of tf.Print. Note that tf.print returns a no-output operator that directly prints the output. Outside of defuns or eager mode, this operator will not be executed unless it is directly specified in session.run or used as a control dependency for other operators. This is only a concern in graph mode. Below is an example of how to ensure tf.print executes in graph mode:\n')
@tf_export(v1=['Print'])
@dispatch.add_dispatch_support
def Print(input_, data, message=None, first_n=None, summarize=None, name=None):
    """Prints a list of tensors.

  This is an identity op (behaves like `tf.identity`) with the side effect
  of printing `data` when evaluating.

  Note: This op prints to the standard error. It is not currently compatible
    with jupyter notebook (printing to the notebook *server's* output, not into
    the notebook).

  @compatibility(TF2)
  This API is deprecated. Use `tf.print` instead. `tf.print` does not need the
  `input_` argument.

  `tf.print` works in TF2 when executing eagerly and inside a `tf.function`.

  In TF1-styled sessions, an explicit control dependency declaration is needed
  to execute the `tf.print` operation. Refer to the documentation of
  `tf.print` for more details.
  @end_compatibility

  Args:
    input_: A tensor passed through this op.
    data: A list of tensors to print out when op is evaluated.
    message: A string, prefix of the error message.
    first_n: Only log `first_n` number of times. Negative numbers log always;
      this is the default.
    summarize: Only print this many entries of each tensor. If None, then a
      maximum of 3 elements are printed per input tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type and contents as `input_`.

    ```python
    sess = tf.compat.v1.Session()
    with sess.as_default():
        tensor = tf.range(10)
        print_op = tf.print(tensor)
        with tf.control_dependencies([print_op]):
          out = tf.add(tensor, tensor)
        sess.run(out)
    ```
  """
    return gen_logging_ops._print(input_, data, message, first_n, summarize, name)