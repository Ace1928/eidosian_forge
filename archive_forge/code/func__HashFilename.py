from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import hashlib
import json
import os
import re
import sys
import six
from boto import config
from gslib.exception import CommandException
from gslib.utils.boto_util import GetGsutilStateDir
from gslib.utils.boto_util import ResumableThreshold
from gslib.utils.constants import UTF8
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.system_util import CreateDirIfNeeded
def _HashFilename(filename):
    """Apply a hash function (SHA1) to shorten the passed file name.

  The spec for the hashed file name is as follows:

      TRACKER_<hash>_<trailing>

  where hash is a SHA1 hash on the original file name and trailing is
  the last 16 chars from the original file name. Max file name lengths
  vary by operating system so the goal of this function is to ensure
  the hashed version takes fewer than 100 characters.

  Args:
    filename: file name to be hashed. May be unicode or bytes.

  Returns:
    shorter, hashed version of passed file name
  """
    if isinstance(filename, six.text_type):
        filename_bytes = filename.encode(UTF8)
        filename_str = filename
    else:
        filename_bytes = filename
        filename_str = filename.decode(UTF8)
    m = hashlib.sha1(filename_bytes)
    return 'TRACKER_' + m.hexdigest() + '.' + filename_str[-16:]