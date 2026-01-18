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
def GetSlicedDownloadTrackerFilePaths(dst_url, api_selector, num_components=None):
    """Gets a list of sliced download tracker file paths.

  The list consists of the parent tracker file path in index 0, and then
  any existing component tracker files in [1:].

  Args:
    dst_url: Destination URL for tracker file.
    api_selector: API to use for this operation.
    num_components: The number of component tracker files, if already known.
                    If not known, the number will be retrieved from the parent
                    tracker file on disk.
  Returns:
    File path to tracker file.
  """
    parallel_tracker_file_path = GetTrackerFilePath(dst_url, TrackerFileType.SLICED_DOWNLOAD, api_selector)
    tracker_file_paths = [parallel_tracker_file_path]
    if num_components is None:
        tracker_file = None
        try:
            tracker_file = open(parallel_tracker_file_path, 'r')
            num_components = json.load(tracker_file)['num_components']
        except (IOError, ValueError):
            return tracker_file_paths
        finally:
            if tracker_file:
                tracker_file.close()
    for i in range(num_components):
        tracker_file_paths.append(GetTrackerFilePath(dst_url, TrackerFileType.DOWNLOAD_COMPONENT, api_selector, component_num=i))
    return tracker_file_paths