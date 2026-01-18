from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import namedtuple
import errno
import json
import random
import six
import gslib
from gslib.exception import CommandException
from gslib.tracker_file import (WriteJsonDataToTrackerFile,
from gslib.utils.constants import UTF8
def WriteComponentToParallelUploadTrackerFile(tracker_file_name, tracker_file_lock, component, logger, encryption_key_sha256=None):
    """Rewrites an existing tracker file with info about the uploaded component.

  Follows the format described in _CreateParallelUploadTrackerFile.

  Args:
    tracker_file_name: Tracker file to append to.
    tracker_file_lock: Thread and process-safe Lock protecting the tracker file.
    component: ObjectFromTracker describing the object that was uploaded.
    logger: logging.Logger for outputting log messages.
    encryption_key_sha256: Encryption key SHA256 for use in this upload, if any.
  """
    with tracker_file_lock:
        existing_enc_key_sha256, prefix, existing_components = ReadParallelUploadTrackerFile(tracker_file_name, logger)
        if existing_enc_key_sha256 != encryption_key_sha256:
            raise CommandException('gsutil client error: encryption key SHA256 (%s) in tracker file does not match encryption key SHA256 (%s) of component %s' % (existing_enc_key_sha256, encryption_key_sha256, component.object_name))
        newly_completed_components = [component]
        completed_components = existing_components + newly_completed_components
        WriteParallelUploadTrackerFile(tracker_file_name, prefix, completed_components, encryption_key_sha256=encryption_key_sha256)