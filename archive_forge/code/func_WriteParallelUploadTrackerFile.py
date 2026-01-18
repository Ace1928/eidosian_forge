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
def WriteParallelUploadTrackerFile(tracker_file_name, prefix, components, encryption_key_sha256=None):
    """Writes information about components that were successfully uploaded.

  The tracker file is serialized JSON of the form:
  {
    "encryption_key_sha256": sha256 hash of encryption key (or null),
    "prefix": Prefix used for the component objects,
    "components": [
      {
       "component_name": Component object name,
       "component_generation": Component object generation (or null),
      }, ...
    ]
  }
  where N is the number of components that have been successfully uploaded.

  This function is not thread-safe and must be protected by a lock if
  called within Command.Apply.

  Args:
    tracker_file_name: The name of the parallel upload tracker file.
    prefix: The generated prefix that used for uploading any existing
        components.
    components: A list of ObjectFromTracker objects that were uploaded.
    encryption_key_sha256: Encryption key SHA256 for use in this upload, if any.
  """
    if six.PY3:
        if isinstance(encryption_key_sha256, bytes):
            encryption_key_sha256 = encryption_key_sha256.decode('ascii')
    tracker_components = []
    for component in components:
        tracker_components.append({_CompositeUploadTrackerEntry.COMPONENT_NAME: component.object_name, _CompositeUploadTrackerEntry.COMPONENT_GENERATION: component.generation})
    tracker_file_data = {_CompositeUploadTrackerEntry.COMPONENTS_LIST: tracker_components, _CompositeUploadTrackerEntry.ENC_SHA256: encryption_key_sha256, _CompositeUploadTrackerEntry.PREFIX: prefix}
    WriteJsonDataToTrackerFile(tracker_file_name, tracker_file_data)