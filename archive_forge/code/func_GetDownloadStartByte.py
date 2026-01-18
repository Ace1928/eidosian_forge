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
def GetDownloadStartByte(src_obj_metadata, dst_url, api_selector, start_byte, existing_file_size, component_num=None):
    """Returns the download starting point.

  The methodology of this function is the same as in
  ReadOrCreateDownloadTrackerFile, with the difference that we are not
  interested here in possibly creating a tracker file. In case there is no
  tracker file, this means the download starting point is start_byte.

  Args:
    src_obj_metadata: Metadata for the source object. Must include etag and
                      generation.
    dst_url: Destination URL for tracker file.
    api_selector: API to use for this operation.
    start_byte: The start byte of the byte range for this download.
    existing_file_size: Size of existing file for this download on disk.
    component_num: The component number, if this is a component of a parallel
                   download, else None.

  Returns:
    download_start_byte: The first byte that still needs to be downloaded.
  """
    assert src_obj_metadata.etag
    tracker_file_name = None
    if src_obj_metadata.size < ResumableThreshold():
        return start_byte
    if component_num is None:
        tracker_file_type = TrackerFileType.DOWNLOAD
    else:
        tracker_file_type = TrackerFileType.DOWNLOAD_COMPONENT
    tracker_file_name = GetTrackerFilePath(dst_url, tracker_file_type, api_selector, component_num=component_num)
    tracker_file = None
    try:
        tracker_file = open(tracker_file_name, 'r')
        if tracker_file_type is TrackerFileType.DOWNLOAD:
            etag_value = tracker_file.readline().rstrip('\n')
            if etag_value == src_obj_metadata.etag:
                return existing_file_size
        elif tracker_file_type is TrackerFileType.DOWNLOAD_COMPONENT:
            component_data = json.loads(tracker_file.read())
            if component_data['etag'] == src_obj_metadata.etag and component_data['generation'] == src_obj_metadata.generation:
                return component_data['download_start_byte']
    except (IOError, ValueError):
        pass
    finally:
        if tracker_file:
            tracker_file.close()
    return start_byte