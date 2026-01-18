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
def CreateTempDir(self, test_files=0, contents=None):
    """Creates a temporary directory on disk.

    The directory and all of its contents will be deleted after the test.

    Args:
      test_files: The number of test files to place in the directory or a list
                  of test file names.
      contents: The contents for each generated test file.

    Returns:
      The path to the new temporary directory.
    """
    tmpdir = tempfile.mkdtemp(prefix=self.MakeTempName('directory'))
    self.tempdirs.append(tmpdir)
    try:
        iter(test_files)
    except TypeError:
        test_files = [self.MakeTempName('file') for _ in range(test_files)]
    for i, name in enumerate(test_files):
        contents_file = contents
        if contents_file is None:
            contents_file = ('test %d' % i).encode('ascii')
        self.CreateTempFile(tmpdir=tmpdir, file_name=name, contents=contents_file)
    return tmpdir