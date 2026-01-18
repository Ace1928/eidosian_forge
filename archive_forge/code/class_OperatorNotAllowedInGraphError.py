import traceback
import warnings
from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python import _pywrap_py_exception_registry
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export('errors.OperatorNotAllowedInGraphError', v1=[])
class OperatorNotAllowedInGraphError(TypeError):
    """Raised when an unsupported operator is present in Graph execution.

  For example, using a `tf.Tensor` as a Python `bool` inside a Graph will
  raise `OperatorNotAllowedInGraphError`. Iterating over values inside a
  `tf.Tensor` is also not supported in Graph execution.

  Example:
  >>> @tf.function
  ... def iterate_over(t):
  ...   a,b,c = t
  ...   return a
  >>>
  >>> iterate_over(tf.constant([1, 2, 3]))
  Traceback (most recent call last):
  ...
  OperatorNotAllowedInGraphError: ...

  """
    pass