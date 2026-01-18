from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from functools import wraps
import os.path
import random
import re
import shutil
import tempfile
import six
import boto
import gslib.tests.util as util
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils.posix_util import NA_ID
from gslib.utils.posix_util import NA_MODE
def CreateTempFile(self, tmpdir=None, contents=None, file_name=None, mtime=None, mode=NA_MODE, uid=NA_ID, gid=NA_ID):
    """Creates a temporary file on disk.

    Note: if mode, uid, or gid are present, they must be validated by
    ValidateFilePermissionAccess and ValidatePOSIXMode before calling this
    function.

    Args:
      tmpdir: The temporary directory to place the file in. If not specified, a
              new temporary directory is created.
      contents: The contents to write to the file. If not specified, a test
                string is constructed and written to the file. Since the file
                is opened 'wb', the contents must be bytes.
      file_name: The name to use for the file. If not specified, a temporary
                 test file name is constructed. This can also be a tuple, where
                 ('dir', 'foo') means to create a file named 'foo' inside a
                 subdirectory named 'dir'.
      mtime: The modification time of the file in POSIX time (seconds since
             UTC 1970-01-01). If not specified, this defaults to the current
             system time.
      mode: The POSIX mode for the file. Must be a base-8 3-digit integer
            represented as a string.
      uid: A POSIX user ID.
      gid: A POSIX group ID.

    Returns:
      The path to the new temporary file.
    """
    tmpdir = six.ensure_str(tmpdir or self.CreateTempDir())
    file_name = file_name or self.MakeTempName(str('file'))
    if isinstance(file_name, (six.text_type, six.binary_type)):
        fpath = os.path.join(tmpdir, six.ensure_str(file_name))
    else:
        file_name = map(six.ensure_str, file_name)
        fpath = os.path.join(tmpdir, *file_name)
    if not os.path.isdir(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))
    if isinstance(fpath, six.binary_type):
        fpath = fpath.decode(UTF8)
    with open(fpath, 'wb') as f:
        contents = contents if contents is not None else self.MakeTempName(str('contents'))
        if isinstance(contents, bytearray):
            contents = bytes(contents)
        else:
            contents = six.ensure_binary(contents)
        f.write(contents)
    if mtime is not None:
        os.utime(fpath, (mtime, mtime))
    if uid != NA_ID or int(gid) != NA_ID:
        os.chown(fpath, uid, int(gid))
    if int(mode) != NA_MODE:
        os.chmod(fpath, int(mode, 8))
    return fpath