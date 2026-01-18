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
def WriteFileAtomically(file_name, contents, convert_invalid_windows_characters=False):
    """Writes a file to disk safely cross platform.

  Specified directories will be created if they don't exist.

  Writes a file to disk safely cross platform. Note that on Windows, there
  is no good way to atomically write a file to disk.

  Args:
    file_name: The actual file to write to.
    contents:  The file contents to write.
    convert_invalid_windows_characters: bool, Convert invalid Windows path
        characters with an 'unsupported' symbol rather than trigger an OSError
        on Windows (e.g. "file|.txt" -> "file$.txt").

  Raises:
    ValueError: file_name or contents is empty.
    TypeError: contents is not a valid string.
  """
    if not file_name or contents is None:
        raise ValueError('Empty file_name [{}] or contents [{}].'.format(file_name, contents))
    if not isinstance(contents, six.string_types):
        raise TypeError('Invalid contents [{}].'.format(contents))
    dirname = os.path.dirname(file_name)
    try:
        os.makedirs(dirname)
    except os.error:
        pass
    if platforms.OperatingSystem.IsWindows():
        WriteFileContents(file_name, contents, private=True, convert_invalid_windows_characters=convert_invalid_windows_characters)
    else:
        with tempfile.NamedTemporaryFile(mode='w', dir=dirname, delete=False) as temp_file:
            temp_file.write(contents)
            temp_file.flush()
            os.rename(temp_file.name, file_name)