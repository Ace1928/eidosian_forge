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
def ReadRewriteTrackerFile(tracker_file_name, rewrite_params_hash):
    """Attempts to read a rewrite tracker file.

  Args:
    tracker_file_name: Tracker file path string.
    rewrite_params_hash: MD5 hex digest of rewrite call parameters constructed
        by HashRewriteParameters.

  Returns:
    String rewrite_token for resuming rewrite requests if a matching tracker
    file exists, None otherwise (which will result in starting a new rewrite).
  """
    tracker_file = None
    if not rewrite_params_hash:
        return
    try:
        tracker_file = open(tracker_file_name, 'r')
        existing_hash = tracker_file.readline().rstrip('\n')
        if existing_hash == rewrite_params_hash:
            return tracker_file.readline().rstrip('\n')
    except IOError as e:
        if e.errno != errno.ENOENT:
            sys.stderr.write("Couldn't read Copy tracker file (%s): %s. Restarting copy from scratch." % (tracker_file_name, e.strerror))
    finally:
        if tracker_file:
            tracker_file.close()