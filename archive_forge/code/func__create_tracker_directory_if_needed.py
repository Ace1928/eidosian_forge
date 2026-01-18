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
def _create_tracker_directory_if_needed():
    """Looks up or creates the gcloud storage tracker file directory.

  Resumable transfer tracker files will be kept here.

  Returns:
    The path string to the tracker directory.
  """
    tracker_directory = properties.VALUES.storage.tracker_files_directory.Get()
    files.MakeDir(tracker_directory)
    return tracker_directory