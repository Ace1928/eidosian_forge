from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import enum
import errno
import hashlib
import io
import logging
import os
import shutil
import stat
import sys
import tempfile
import time
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding as encoding_util
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import retry
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _ShouldRetryOperation(func, exc_info):
    """Matches specific error types that should be retried.

  This will retry the following errors:
    WindowsError(5, 'Access is denied'), When trying to delete a readonly file
    WindowsError(32, 'The process cannot access the file because it is being '
      'used by another process'), When a file is in use.
    WindowsError(145, 'The directory is not empty'), When a directory cannot be
      deleted.

  Args:
    func: function, The function that failed.
    exc_info: sys.exc_info(), The current exception state.

  Returns:
    True if the error can be retried or false if we should just fail.
  """
    if not (func == os.remove or func == os.rmdir or func == os.unlink):
        return False
    if not WindowsError:
        return False
    e = exc_info[1]
    return getattr(e, 'winerror', None) in RETRY_ERROR_CODES