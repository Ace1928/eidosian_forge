from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import os
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import fast_crc32c_util
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import path_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.tasks import patch_file_posix_task
from googlecloudsdk.command_lib.storage.tasks.cp import copy_task_factory
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.objects import patch_object_task
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def _get_url_string_minus_base_container(object_resource, container_resource):
    """Removes container URL prefix from object URL."""
    container_url = container_resource.storage_url
    container_url_string_with_trailing_delimiter = container_url.join('').versionless_url_string
    object_url_string = object_resource.storage_url.versionless_url_string
    if not object_url_string.startswith(container_url_string_with_trailing_delimiter):
        raise errors.Error('Received container {} that does not contain object {}.'.format(container_url_string_with_trailing_delimiter, object_url_string))
    return object_url_string[len(container_url_string_with_trailing_delimiter):]