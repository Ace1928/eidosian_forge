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
def _write_json_to_tracker_file(tracker_file_path, data):
    """Creates a tracker file and writes JSON to it.

  Args:
    tracker_file_path (str): The path to the tracker file.
    data (object): JSON-serializable data to write to file.
  """
    json_string = json.dumps(data)
    _write_tracker_file(tracker_file_path, json_string)