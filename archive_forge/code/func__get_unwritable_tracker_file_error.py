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
def _get_unwritable_tracker_file_error(error, tracker_file_path):
    """Edits error to use custom unwritable message.

  Args:
    error (Exception): Python error to modify message of.
    tracker_file_path (str): Tracker file path there were issues writing to.

  Returns:
    Exception argument with altered error message.
  """
    original_error_text = getattr(error, 'strerror')
    if not original_error_text:
        original_error_text = '[No strerror]'
    return type(error)('Could not write tracker file ({}): {}. This can happen if gcloud storage is configured to save tracker files to an unwritable directory.'.format(tracker_file_path, original_error_text))