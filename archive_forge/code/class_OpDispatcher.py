import collections
import itertools
import typing  # pylint: disable=unused-import (used in doctests)
from tensorflow.python.framework import _pywrap_python_api_dispatcher as _api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export as tf_export_lib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import type_annotations
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.dispatch.OpDispatcher', v1=[])
class OpDispatcher(object):
    """Abstract base class for TensorFlow operator dispatchers.

  Each operation dispatcher acts as an override handler for a single
  TensorFlow operation, and its results are used when the handler indicates
  that it can handle the operation's arguments (by returning any value other
  than `OpDispatcher.NOT_SUPPORTED`).
  """
    NOT_SUPPORTED = object()

    def handle(self, args, kwargs):
        """Handle this dispatcher's operation with the specified arguments.

    If this operation dispatcher can handle the given arguments, then
    return an appropriate value (or raise an appropriate exception).

    Args:
      args: The arguments to the operation.
      kwargs: They keyword arguments to the operation.

    Returns:
      The result of the operation, or `OpDispatcher.NOT_SUPPORTED` if this
      dispatcher can not handle the given arguments.
    """
        return self.NOT_SUPPORTED

    def register(self, op):
        """Register this dispatcher as a handler for `op`.

    Args:
      op: Python function: the TensorFlow operation that should be handled. Must
        have a dispatch list (which is added automatically for generated ops,
        and can be added to Python ops using the `add_dispatch_support`
        decorator).
    """
        if not hasattr(op, FALLBACK_DISPATCH_ATTR):
            raise AssertionError('Dispatching not enabled for %s' % op)
        getattr(op, FALLBACK_DISPATCH_ATTR).append(self)