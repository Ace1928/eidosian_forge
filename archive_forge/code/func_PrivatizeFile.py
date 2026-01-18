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
def PrivatizeFile(path):
    """Makes an existing file private or creates a new, empty private file.

  In theory it would be better to return the open file descriptor so that it
  could be used directly. The issue that we would need to pass an encoding to
  os.fdopen() and on Python 2. This is not supported. Instead we just create
  the empty file and then we will just open it normally later to do the write.

  Args:
    path: str, The path of the file to create or privatize.
  """
    try:
        if os.path.exists(path):
            os.chmod(path, 384)
        else:
            _MakePathToFile(path, mode=448)
            flags = os.O_RDWR | os.O_CREAT | os.O_TRUNC
            if hasattr(os, 'O_NOINHERIT'):
                flags |= os.O_NOINHERIT
            fd = os.open(path, flags, 384)
            os.close(fd)
    except EnvironmentError as e:
        raise Error('Unable to create private file [{0}]: {1}'.format(path, e))