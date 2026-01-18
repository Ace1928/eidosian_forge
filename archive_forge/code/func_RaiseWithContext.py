from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import traceback
from googlecloudsdk.core.util import encoding
import six
def RaiseWithContext(orig_exc_type, orig_exc, orig_exc_trace, curr_exc_type, curr_exc, curr_exc_trace):
    """Raises an exception that occurs when handling another exception.

  Python 3 does this sort of exception chaining natively, but it's not supported
  in Python 2. So when running in Python 2, we manually reproduce the error
  message we would get it in Python 3. It won't look identical but it's good
  enough for debugging purposes so that we don't lose the exception context.

  Args:
    orig_exc_type: The type of the original exception being handled.
    orig_exc: The original exception.
    orig_exc_trace: The traceback of the original exception.
    curr_exc_type: The type of the current exception being handled.
    curr_exc: The current exception.
    curr_exc_trace: The traceback of the current exception.

  Raises:
    Exception: The current exception with added context from the original
      exception being handled.
  """
    if not six.PY2 or not orig_exc:
        raise curr_exc
    orig_exc_msg = _FormatException(orig_exc_type, orig_exc, orig_exc_trace)
    curr_exc_msg = _FormatException(curr_exc_type, curr_exc, curr_exc_trace)
    new_exc_msg = '\n\n{}\nDuring handling of the above exception, another exception occurred:\n\n{}'.format(orig_exc_msg, curr_exc_msg)
    six.reraise(curr_exc_type, curr_exc_type(new_exc_msg), orig_exc_trace)