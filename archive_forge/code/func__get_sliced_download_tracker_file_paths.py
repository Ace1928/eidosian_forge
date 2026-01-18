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
def _get_sliced_download_tracker_file_paths(destination_url):
    """Gets a list of tracker file paths for each slice of a sliced download.

  The returned list consists of the parent tracker file path in index 0
  followed by component tracker files.

  Args:
    destination_url: Destination URL for tracker file.

  Returns:
    List of string file paths to tracker files.
  """
    parallel_tracker_file_path = get_tracker_file_path(destination_url, TrackerFileType.SLICED_DOWNLOAD)
    tracker_file_paths = [parallel_tracker_file_path]
    tracker_file = None
    try:
        tracker_file = files.FileReader(parallel_tracker_file_path)
        total_components = json.load(tracker_file)['total_components']
    except files.MissingFileError:
        return tracker_file_paths
    finally:
        if tracker_file:
            tracker_file.close()
    for i in range(total_components):
        tracker_file_paths.append(get_tracker_file_path(destination_url, TrackerFileType.DOWNLOAD_COMPONENT, component_number=i))
    return tracker_file_paths