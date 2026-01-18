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
def _FileOpener(path, mode, verb, encoding=None, private=False, create_path=False, newline=None, convert_invalid_windows_characters=False):
    """Opens a file in various modes and does error handling."""
    if convert_invalid_windows_characters and platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS:
        path = platforms.MakePathWindowsCompatible(path)
    if private:
        PrivatizeFile(path)
    if create_path:
        _MakePathToFile(path)
    try:
        return io.open(path, mode, encoding=encoding, newline=newline)
    except EnvironmentError as e:
        exc_type = Error
        if isinstance(e, IOError) and e.errno == errno.ENOENT:
            exc_type = MissingFileError
        raise exc_type('Unable to {0} file [{1}]: {2}'.format(verb, path, e))