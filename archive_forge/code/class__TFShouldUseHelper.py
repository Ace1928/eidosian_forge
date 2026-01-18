import copy
import sys
import textwrap
import traceback
import types
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_decorator
class _TFShouldUseHelper(object):
    """Object stored in TFShouldUse-wrapped objects.

  When it is deleted it will emit a warning or error if its `sate` method
  has not been called by time of deletion, and Tensorflow is not executing
  eagerly or inside a tf.function (which use autodeps and resolve the
  main issues this wrapper warns about).
  """

    def __init__(self, type_, repr_, stack_frame, error_in_function, warn_in_eager):
        self._type = type_
        self._repr = repr_
        self._stack_frame = stack_frame
        self._error_in_function = error_in_function
        if context.executing_eagerly():
            self._sated = not warn_in_eager
        elif ops.inside_function():
            if error_in_function:
                self._sated = False
                ops.add_exit_callback_to_default_func_graph(lambda: self._check_sated(raise_error=True))
            else:
                self._sated = True
        else:
            self._sated = False

    def sate(self):
        self._sated = True
        self._type = None
        self._repr = None
        self._stack_frame = None
        self._logging_module = None

    def _check_sated(self, raise_error):
        """Check if the object has been sated."""
        if self._sated:
            return
        creation_stack = ''.join([line.rstrip() for line in traceback.format_stack(self._stack_frame, limit=5)])
        if raise_error:
            try:
                raise RuntimeError('Object was never used (type {}): {}.  If you want to mark it as used call its "mark_used()" method.  It was originally created here:\n{}'.format(self._type, self._repr, creation_stack))
            finally:
                self.sate()
        else:
            tf_logging.error('==================================\nObject was never used (type {}):\n{}\nIf you want to mark it as used call its "mark_used()" method.\nIt was originally created here:\n{}\n=================================='.format(self._type, self._repr, creation_stack))

    def __del__(self):
        self._check_sated(raise_error=False)