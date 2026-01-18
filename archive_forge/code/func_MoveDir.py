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
def MoveDir(src, dst):
    """Recursively moves a directory to another location.

  This code is mostly copied from shutil.move(), but has been scoped down to
  specifically handle only directories.  The src must be a directory, and
  the dst must not exist.  It uses functions from this module to be resilient
  against spurious file system errors in Windows.  It will try to do an
  os.rename() of the directory.  If that fails, the tree will be copied to the
  new location and then deleted from the old location.

  Args:
    src: str, The directory path to move.
    dst: str, The path to move the directory to.

  Raises:
    Error: If the src or dst directories are not valid.
  """
    if not os.path.isdir(src):
        raise Error("Source path '{0}' must be a directory".format(src))
    if os.path.exists(dst):
        raise Error("Destination path '{0}' already exists".format(dst))
    if _DestInSrc(src, dst):
        raise Error("Cannot move a directory '{0}' into itself '{1}'.".format(src, dst))
    try:
        logging.debug('Attempting to move directory [%s] to [%s]', src, dst)
        try:
            os.rename(src, dst)
        except OSError:
            if not _RetryOperation(sys.exc_info(), os.rename, (src, dst)):
                raise
    except OSError as e:
        logging.debug('Directory rename failed.  Falling back to copy. [%s]', e)
        shutil.copytree(src, dst, symlinks=True)
        RmTree(src)