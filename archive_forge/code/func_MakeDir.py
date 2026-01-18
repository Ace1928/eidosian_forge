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
def MakeDir(path, mode=511, convert_invalid_windows_characters=False):
    """Creates the given directory and its parents and does not fail if it exists.

  Args:
    path: str, The path of the directory to create.
    mode: int, The permissions to give the created directories. 0777 is the
      default mode for os.makedirs(), allowing reading, writing, and listing by
      all users on the machine.
    convert_invalid_windows_characters: bool, Convert invalid Windows path
      characters with an 'unsupported' symbol rather than trigger an OSError on
      Windows (e.g. "file|.txt" -> "file$.txt").

  Raises:
    Error: if the operation fails and we can provide extra information.
    OSError: if the operation fails.
  """
    if convert_invalid_windows_characters and platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS:
        path = platforms.MakePathWindowsCompatible(path)
    try:
        os.makedirs(path, mode=mode)
    except OSError as ex:
        base_msg = 'Could not create directory [{0}]: '.format(path)
        if ex.errno == errno.EEXIST and os.path.isdir(path):
            pass
        elif ex.errno == errno.EEXIST and os.path.isfile(path):
            raise Error(base_msg + 'A file exists at that location.\n\n')
        elif ex.errno == errno.EACCES:
            raise Error(base_msg + 'Permission denied.\n\n' + 'Please verify that you have permissions to write to the parent directory.')
        else:
            raise