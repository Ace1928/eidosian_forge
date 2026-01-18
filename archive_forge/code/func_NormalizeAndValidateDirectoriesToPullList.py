from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import posixpath
import random
import re
import string
import sys
from googlecloudsdk.api_lib.firebase.test import exceptions as test_exceptions
from googlecloudsdk.api_lib.firebase.test import util as util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import files
import six
def NormalizeAndValidateDirectoriesToPullList(dirs):
    """Validate list of file paths for 'directories-to-pull' flag.

  Also collapse paths to remove "." ".." and "//".

  Args:
    dirs: list of directory names to pull from the device.
  """
    if dirs:
        dirs[:] = [posixpath.abspath(path) if path else path for path in dirs]
    for file_path in dirs or []:
        if not _DIRECTORIES_TO_PULL_PATH_REGEX.match(file_path):
            raise test_exceptions.InvalidArgException('directories_to_pull', 'Invalid path [{0}]'.format(file_path))