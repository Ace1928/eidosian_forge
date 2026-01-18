from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import hashlib
import json
import os
import re
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import scaled_integer
def _get_hashed_tracker_file_path(tracker_file_name, tracker_file_type, resumable_tracker_directory, component_number):
    """Hashes and returns a tracker file path.

  Args:
    tracker_file_name (str): The tracker file name prior to it being hashed.
    tracker_file_type (TrackerFileType): The TrackerFileType of
      res_tracker_file_name.
    resumable_tracker_directory (str): Path to directory of tracker files.
    component_number (int|None): The number of the component is being tracked
      for a sliced download or composite upload.

  Returns:
    Final (hashed) tracker file path.

  Raises:
    Error: Hashed file path is too long.
  """
    hashed_tracker_file_name = get_hashed_file_name(tracker_file_name)
    tracker_file_name_with_type = '{}_TRACKER_{}'.format(tracker_file_type.value.lower(), hashed_tracker_file_name)
    if component_number is None:
        final_tracker_file_name = tracker_file_name_with_type
    else:
        final_tracker_file_name = tracker_file_name_with_type + '_{}'.format(component_number)
    raise_exceeds_max_length_error(final_tracker_file_name)
    tracker_file_path = os.path.join(resumable_tracker_directory, final_tracker_file_name)
    return tracker_file_path