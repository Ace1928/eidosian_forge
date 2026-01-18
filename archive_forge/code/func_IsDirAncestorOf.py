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
def IsDirAncestorOf(ancestor_directory, path):
    """Returns whether ancestor_directory is an ancestor of path.

  Args:
    ancestor_directory: str, path to the directory that is the potential
      ancestor of path
    path: str, path to the file/directory that is a potential descendant of
      ancestor_directory

  Returns:
    bool, whether path has ancestor_directory as an ancestor.

  Raises:
    ValueError: if the given ancestor_directory is not, in fact, a directory.
  """
    if not os.path.isdir(ancestor_directory):
        raise ValueError('[{0}] is not a directory.'.format(ancestor_directory))
    path = encoding_util.Decode(os.path.realpath(path))
    ancestor_directory = encoding_util.Decode(os.path.realpath(ancestor_directory))
    try:
        rel = os.path.relpath(path, ancestor_directory)
    except ValueError:
        return False
    return not rel.startswith('..' + os.path.sep) and rel != '..'