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
def ReadOrCreateDownloadTrackerFile(src_obj_metadata, dst_url, logger, api_selector, start_byte, existing_file_size, component_num=None):
    """Checks for a download tracker file and creates one if it does not exist.

  The methodology for determining the download start point differs between
  normal and sliced downloads. For normal downloads, the existing bytes in
  the file are presumed to be correct and have been previously downloaded from
  the server (if a tracker file exists). In this case, the existing file size
  is used to determine the download start point. For sliced downloads, the
  number of bytes previously retrieved from the server cannot be determined
  from the existing file size, and so the number of bytes known to have been
  previously downloaded is retrieved from the tracker file.

  Args:
    src_obj_metadata: Metadata for the source object. Must include etag and
                      generation.
    dst_url: Destination URL for tracker file.
    logger: For outputting log messages.
    api_selector: API to use for this operation.
    start_byte: The start byte of the byte range for this download.
    existing_file_size: Size of existing file for this download on disk.
    component_num: The component number, if this is a component of a parallel
                   download, else None.

  Returns:
    tracker_file_name: The name of the tracker file, if one was used.
    download_start_byte: The first byte that still needs to be downloaded.
  """
    assert src_obj_metadata.etag
    tracker_file_name = None
    if src_obj_metadata.size < ResumableThreshold():
        return (tracker_file_name, start_byte)
    download_name = dst_url.object_name
    if component_num is None:
        tracker_file_type = TrackerFileType.DOWNLOAD
    else:
        tracker_file_type = TrackerFileType.DOWNLOAD_COMPONENT
        download_name += ' component %d' % component_num
    tracker_file_name = GetTrackerFilePath(dst_url, tracker_file_type, api_selector, component_num=component_num)
    tracker_file = None
    try:
        tracker_file = open(tracker_file_name, 'r')
        if tracker_file_type is TrackerFileType.DOWNLOAD:
            etag_value = tracker_file.readline().rstrip('\n')
            if etag_value == src_obj_metadata.etag:
                return (tracker_file_name, existing_file_size)
        elif tracker_file_type is TrackerFileType.DOWNLOAD_COMPONENT:
            component_data = json.loads(tracker_file.read())
            if component_data['etag'] == src_obj_metadata.etag and component_data['generation'] == src_obj_metadata.generation:
                return (tracker_file_name, component_data['download_start_byte'])
        logger.warn("Tracker file doesn't match for download of %s. Restarting download from scratch." % download_name)
    except (IOError, ValueError) as e:
        if isinstance(e, ValueError) or e.errno != errno.ENOENT:
            logger.warn("Couldn't read download tracker file (%s): %s. Restarting download from scratch." % (tracker_file_name, str(e)))
    finally:
        if tracker_file:
            tracker_file.close()
    if tracker_file_type is TrackerFileType.DOWNLOAD:
        _WriteTrackerFile(tracker_file_name, '%s\n' % src_obj_metadata.etag)
    elif tracker_file_type is TrackerFileType.DOWNLOAD_COMPONENT:
        WriteDownloadComponentTrackerFile(tracker_file_name, src_obj_metadata, start_byte)
    return (tracker_file_name, start_byte)