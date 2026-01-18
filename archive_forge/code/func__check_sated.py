import copy
import sys
import textwrap
import traceback
import types
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_decorator
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