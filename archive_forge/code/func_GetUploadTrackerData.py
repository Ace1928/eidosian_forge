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
def GetUploadTrackerData(tracker_file_name, logger, encryption_key_sha256=None):
    """Reads tracker data from an upload tracker file if it exists.

  Deletes the tracker file if it uses an old format or the desired
  encryption key has changed.

  Args:
    tracker_file_name: Tracker file name for this upload.
    logger: logging.Logger for outputting log messages.
    encryption_key_sha256: Encryption key SHA256 for use in this upload, if any.

  Returns:
    Serialization data if the tracker file already exists (resume existing
    upload), None otherwise.
  """
    tracker_file = None
    remove_tracker_file = False
    encryption_restart = False
    try:
        tracker_file = open(tracker_file_name, 'r')
        tracker_data = tracker_file.read()
        tracker_json = json.loads(tracker_data)
        if tracker_json[ENCRYPTION_UPLOAD_TRACKER_ENTRY] != encryption_key_sha256:
            encryption_restart = True
            remove_tracker_file = True
        else:
            return tracker_json[SERIALIZATION_UPLOAD_TRACKER_ENTRY]
    except IOError as e:
        if e.errno != errno.ENOENT:
            logger.warn("Couldn't read upload tracker file (%s): %s. Restarting upload from scratch.", tracker_file_name, e.strerror)
    except (KeyError, ValueError) as e:
        remove_tracker_file = True
        if encryption_key_sha256 is not None:
            encryption_restart = True
        else:
            return tracker_data
    finally:
        if tracker_file:
            tracker_file.close()
        if encryption_restart:
            logger.warn('Upload tracker file (%s) does not match current encryption key. Restarting upload from scratch with a new tracker file that uses the current encryption key.', tracker_file_name)
        if remove_tracker_file:
            DeleteTrackerFile(tracker_file_name)