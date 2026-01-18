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
def RmTree(path):
    """Calls shutil.rmtree() with error handling to fix Windows problems.

  It also ensures that the top level directory deletion is actually reflected
  in the file system before this returns.

  Args:
    path: str, The path to remove.
  """
    path = six.text_type(path)
    if sys.version_info[:2] < (3, 12):
        shutil.rmtree(path, onerror=_HandleRemoveError)
    else:
        shutil.rmtree(path, onexc=_HandleRemoveError)
    retries_left = NUM_RETRIES
    while os.path.isdir(path) and retries_left > 0:
        logging.debug('Waiting for directory to disappear: %s', path)
        retries_left -= 1
        _WaitForRetry(retries_left)