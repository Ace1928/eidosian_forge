from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import streaming_download_task
def _get_start_byte(start_byte, source_resource_size):
    """Returns the byte index to start streaming from.

  Gets an absolute start byte for object download API calls.

  Args:
    start_byte (int): The start index entered by the user. Negative values are
      interpreted as offsets from the end of the object.
    source_resource_size (int|None): The size of the source resource.

  Returns:
    int: The byte index to start the object download from.
  """
    if start_byte < 0:
        if abs(start_byte) >= source_resource_size:
            return 0
        return source_resource_size - abs(start_byte)
    return start_byte