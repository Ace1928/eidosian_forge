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
def BinaryFileWriter(path, private=False, mode=BinaryFileWriterMode.TRUNCATE, create_path=False, convert_invalid_windows_characters=False):
    """Opens the given file for binary write for use in a 'with' statement.

  Args:
    path: str, The file path to write to.
    private: bool, True to create or update the file permission to be 0o600.
    mode: BinaryFileWriterMode, Determines how to open file for writing.
    create_path: bool, True to create intermediate directories, if needed.
    convert_invalid_windows_characters: bool, Convert invalid Windows path
        characters with an 'unsupported' symbol rather than trigger an OSError
        on Windows (e.g. "file|.txt" -> "file$7.txt").

  Returns:
    A file-like object opened for write in binary mode.
  """
    return _FileOpener(path, mode.value, 'write', private=private, create_path=create_path, convert_invalid_windows_characters=convert_invalid_windows_characters)