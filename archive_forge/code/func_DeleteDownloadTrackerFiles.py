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
def DeleteDownloadTrackerFiles(dst_url, api_selector):
    """Deletes all tracker files corresponding to an object download.

  Args:
    dst_url: StorageUrl describing the destination file.
    api_selector: The Cloud API implementation used.
  """
    DeleteTrackerFile(GetTrackerFilePath(dst_url, TrackerFileType.DOWNLOAD, api_selector))
    tracker_files = GetSlicedDownloadTrackerFilePaths(dst_url, api_selector)
    for tracker_file in tracker_files:
        DeleteTrackerFile(tracker_file)