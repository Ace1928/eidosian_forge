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
def _complete_destination(self, destination_container, source):
    """Gets a valid copy destination incorporating part of the source's name.

    When given a source file or object and a destination resource that should
    be treated as a container, this function uses the last part of the source's
    name to get an object or file resource representing the copy destination.

    For example: given a source `dir/file` and a destination `gs://bucket/`, the
    destination returned is a resource representing `gs://bucket/file`. Check
    the recursive helper function docstring for details on recursion handling.

    Args:
      destination_container (resource_reference.Resource): The destination
        container.
      source (NameExpansionResult): Represents the source resource and the
        expanded parent url in case of recursion.

    Returns:
      The completed destination, a resource_reference.Resource.
    """
    destination_url = destination_container.storage_url
    source_url = source.resource.storage_url
    if source_url.versionless_url_string != source.expanded_url.versionless_url_string:
        destination_suffix = self._get_destination_suffix_for_recursion(destination_container, source)
    else:
        _, _, url_without_scheme = source_url.versionless_url_string.rpartition(source_url.scheme.value + '://')
        if url_without_scheme.endswith(source_url.delimiter):
            url_without_scheme_and_trailing_delimiter = url_without_scheme[:-len(source_url.delimiter)]
        else:
            url_without_scheme_and_trailing_delimiter = url_without_scheme
        _, _, destination_suffix = url_without_scheme_and_trailing_delimiter.rpartition(source_url.delimiter)
        if url_without_scheme_and_trailing_delimiter != url_without_scheme:
            destination_suffix += source_url.delimiter
    destination_url_prefix = storage_url.storage_url_from_string(destination_url.versionless_url_string.rstrip(destination_url.delimiter))
    new_destination_url = destination_url_prefix.join(destination_suffix)
    return resource_reference.UnknownResource(new_destination_url)