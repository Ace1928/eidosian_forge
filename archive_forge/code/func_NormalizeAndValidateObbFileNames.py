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
def NormalizeAndValidateObbFileNames(obb_files):
    """Confirm that any OBB file names follow the required Android pattern.

  Also expand local paths with "~"

  Args:
    obb_files: list of obb file references. Each one is either a filename on the
      local FS or a gs:// reference.
  """
    if obb_files:
        obb_files[:] = [obb_file if not obb_file or obb_file.startswith(storage_util.GSUTIL_BUCKET_PREFIX) else files.ExpandHomeDir(obb_file) for obb_file in obb_files]
    for obb_file in obb_files or []:
        if not _OBB_FILE_REGEX.match(obb_file):
            raise test_exceptions.InvalidArgException('obb_files', '[{0}] is not a valid OBB file name, which must have the format: (main|patch).<versionCode>.<package.name>.obb'.format(obb_file))