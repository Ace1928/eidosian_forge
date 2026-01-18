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
def BinaryFileReader(path):
    """Opens the given file for binary read for use in a 'with' statement.

  Args:
    path: str, The file path to read from.

  Returns:
    A file-like object opened for read in binary mode.
  """
    return _FileOpener(encoding_util.Encode(path, encoding='utf-8'), 'rb', 'read')