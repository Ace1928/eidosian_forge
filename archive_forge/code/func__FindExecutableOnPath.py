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
def _FindExecutableOnPath(executable, path, pathext):
    """Internal function to a find an executable.

  Args:
    executable: The name of the executable to find.
    path: A list of directories to search separated by 'os.pathsep'.
    pathext: An iterable of file name extensions to use.

  Returns:
    str, the path to a file on `path` with name `executable` + `p` for
      `p` in `pathext`.

  Raises:
    ValueError: invalid input.
  """
    if isinstance(pathext, six.string_types):
        raise ValueError("_FindExecutableOnPath(..., pathext='{0}') failed because pathext must be an iterable of strings, but got a string.".format(pathext))
    for ext in pathext:
        for directory in path.split(os.pathsep):
            directory = directory.strip('"')
            full = os.path.normpath(os.path.join(directory, executable) + ext)
            if os.path.isfile(full) and os.access(full, os.X_OK):
                return full
    return None