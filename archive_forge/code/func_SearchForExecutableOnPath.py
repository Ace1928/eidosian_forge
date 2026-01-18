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
def SearchForExecutableOnPath(executable, path=None):
    """Tries to find all 'executable' in the directories listed in the PATH.

  This is mostly copied from distutils.spawn.find_executable() but with a
  few differences.  It does not check the current directory for the
  executable.  We only want to find things that are actually on the path, not
  based on what the CWD is.  It also returns a list of all matching
  executables.  If there are multiple versions of an executable on the path
  it will return all of them at once.

  Args:
    executable: The name of the executable to find
    path: A path to search.  If none, the system PATH will be used.

  Returns:
    A list of full paths to matching executables or an empty list if none
    are found.
  """
    if not path:
        path = _GetSystemPath()
        if not path:
            return []
    paths = path.split(os.pathsep)
    matching = []
    for p in paths:
        f = os.path.join(p, executable)
        if os.path.isfile(f):
            matching.append(f)
    return matching