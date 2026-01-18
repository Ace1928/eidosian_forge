from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def _create_symlink_directory_if_needed():
    """Looks up or creates the gcloud storage symlink file directory.

  Symlink placeholder files will be kept here.

  Returns:
    The path string to the symlink directory.
  """
    symlink_directory = properties.VALUES.storage.symlink_placeholder_directory.Get()
    files.MakeDir(symlink_directory)
    return symlink_directory