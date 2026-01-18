from within calliope.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
from functools import wraps
import os
import sys
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
import six
def ConvertKnownError(exc):
    """Convert the given exception into an alternate type if it is known.

  Searches backwards through Exception type hierarchy until it finds a match.

  Args:
    exc: Exception, the exception to convert.

  Returns:
    (exception, bool), exception is None if this is not a known type, otherwise
    a new exception that should be logged. The boolean is True if the error
    should be printed, or False to just exit without printing.
  """
    if isinstance(exc, ExitCodeNoError):
        return (exc, False)
    elif isinstance(exc, core_exceptions.Error):
        return (exc, True)
    known_err = None
    classes = [type(exc)]
    processed = set([])
    while classes:
        cls = classes.pop(0)
        processed.add(cls)
        name = _GetExceptionName(cls)
        if name == 'builtins.OSError' and _IsSocketError(exc):
            known_err = core_exceptions.NetworkIssueError
        else:
            known_err = _KNOWN_ERRORS.get(name)
        if known_err:
            break
        bases = [bc for bc in cls.__bases__ if bc not in processed and issubclass(bc, Exception)]
        classes.extend([base for base in bases if base is not Exception])
    if not known_err:
        return (None, True)
    new_exc = known_err(exc)
    return (new_exc, True) if new_exc else (exc, True)