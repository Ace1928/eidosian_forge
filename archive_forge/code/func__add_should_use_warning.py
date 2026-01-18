import copy
import sys
import textwrap
import traceback
import types
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_decorator
def _add_should_use_warning(x, error_in_function=False, warn_in_eager=False):
    """Wraps object x so that if it is never used, a warning is logged.

  Args:
    x: Python object.
    error_in_function: Python bool.  If `True`, a `RuntimeError` is raised
      if the returned value is never used when created during `tf.function`
      tracing.
    warn_in_eager: Python bool. If `True` raise warning if in Eager mode as well
      as graph mode.

  Returns:
    An instance of `TFShouldUseWarningWrapper` which subclasses `type(x)`
    and is a very shallow wrapper for `x` which logs access into `x`.
  """
    if x is None or (isinstance(x, list) and (not x)):
        return x
    if context.executing_eagerly() and (not warn_in_eager):
        return x
    if ops.inside_function() and (not error_in_function):
        return x
    try:
        raise ValueError()
    except ValueError:
        stack_frame = sys.exc_info()[2].tb_frame.f_back
    tf_should_use_helper = _TFShouldUseHelper(type_=type(x), repr_=repr(x), stack_frame=stack_frame, error_in_function=error_in_function, warn_in_eager=warn_in_eager)
    return _get_wrapper(x, tf_should_use_helper)