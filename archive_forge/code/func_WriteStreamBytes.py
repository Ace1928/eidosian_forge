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
def WriteStreamBytes(stream, contents):
    """Write the given bytes to the stream.

  Args:
    stream: The raw stream to write to, usually sys.stdout or sys.stderr.
    contents: A byte string to write to the stream.
  """
    if six.PY2:
        with _FileInBinaryMode(stream):
            stream.write(contents)
            stream.flush()
    else:
        stream.buffer.write(contents)