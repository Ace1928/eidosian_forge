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
def _get_raw_destination(destination_string):
    """Converts self._destination_string to a destination resource.

  Args:
    destination_string (str): A string representing the destination url.

  Returns:
    A resource_reference.Resource. Note that this resource may not be a valid
    copy destination if it is a BucketResource, PrefixResource,
    FileDirectoryResource or UnknownResource.

  Raises:
    InvalidUrlError if the destination url is a cloud provider or if it
    specifies
      a version.
  """
    destination_url = storage_url.storage_url_from_string(destination_string)
    if isinstance(destination_url, storage_url.CloudUrl):
        if destination_url.is_provider():
            raise errors.InvalidUrlError('The cp command does not support provider-only destination URLs.')
        elif destination_url.generation is not None:
            raise errors.InvalidUrlError('The destination argument of the cp command cannot be a version-specific URL ({}).'.format(destination_string))
    raw_destination = _expand_destination_wildcards(destination_string)
    if raw_destination:
        return raw_destination
    return resource_reference.UnknownResource(destination_url)