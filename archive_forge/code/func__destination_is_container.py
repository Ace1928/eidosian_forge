from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import path_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.tasks.cp import copy_task_factory
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _destination_is_container(destination):
    """Returns True is the destination can be treated as a container.

  For a CloudUrl, a container is a bucket or a prefix. If the destination does
  not exist, we determine this based on the delimiter.
  For a FileUrl, A container is an existing dir. For non existing path, we
  return False.

  Args:
    destination (resource_reference.Resource): The destination container.

  Returns:
    bool: True if destination is a valid container.
  """
    try:
        if destination.is_container():
            return True
    except errors.ValueCannotBeDeterminedError:
        pass
    destination_url = destination.storage_url
    if isinstance(destination_url, storage_url.FileUrl):
        return os.path.isdir(destination_url.object_name)
    return destination_url.versionless_url_string.endswith(destination_url.delimiter) or (isinstance(destination_url, storage_url.CloudUrl) and destination_url.is_bucket())