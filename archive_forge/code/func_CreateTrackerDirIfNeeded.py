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
def CreateTrackerDirIfNeeded():
    """Looks up or creates the gsutil tracker file directory.

  This is the configured directory where gsutil keeps its resumable transfer
  tracker files. This function creates it if it doesn't already exist.

  Returns:
    The pathname to the tracker directory.
  """
    tracker_dir = config.get('GSUtil', 'resumable_tracker_dir', os.path.join(GetGsutilStateDir(), 'tracker-files'))
    CreateDirIfNeeded(tracker_dir)
    return tracker_dir