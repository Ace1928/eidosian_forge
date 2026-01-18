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
def CreateTempFifo(self, tmpdir=None, file_name=None):
    """Creates a temporary fifo file on disk. Should not be used on Windows.

    Args:
      tmpdir: The temporary directory to place the file in. If not specified, a
          new temporary directory is created.
      file_name: The name to use for the file. If not specified, a temporary
          test file name is constructed. This can also be a tuple, where
          ('dir', 'foo') means to create a file named 'foo' inside a
          subdirectory named 'dir'.

    Returns:
      The path to the new temporary fifo.
    """
    tmpdir = tmpdir or self.CreateTempDir()
    file_name = file_name or self.MakeTempName('fifo')
    if isinstance(file_name, six.string_types):
        fpath = os.path.join(tmpdir, file_name)
    else:
        fpath = os.path.join(tmpdir, *file_name)
    os.mkfifo(fpath)
    return fpath