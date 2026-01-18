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
def _get_destination_suffix_for_recursion(self, destination_container, source):
    """Returns the suffix required to complete the destination URL.

    Let's assume the following:
      User command => cp -r */base_dir gs://dest/existing_prefix
      source.resource.storage_url => a/base_dir/c/d.txt
      source.expanded_url => a/base_dir
      destination_container.storage_url => gs://dest/existing_prefix

    If the destination container exists, the entire directory gets copied:
    Result => gs://dest/existing_prefix/base_dir/c/d.txt

    Args:
      destination_container (resource_reference.Resource): The destination
        container.
      source (NameExpansionResult): Represents the source resource and the
        expanded parent url in case of recursion.

    Returns:
      (str) The suffix to be appended to the destination container.
    """
    source_prefix_to_ignore = storage_url.rstrip_one_delimiter(source.expanded_url.versionless_url_string, source.expanded_url.delimiter)
    expanded_url_is_valid_parent = _is_expanded_url_valid_parent_dir(source.expanded_url)
    if not expanded_url_is_valid_parent and self._has_multiple_top_level_sources:
        raise errors.InvalidUrlError('Presence of multiple top-level sources and invalid expanded URL make file name conflicts possible for URL: {}'.format(source.resource))
    is_top_level_source_object_name_conflict_possible = isinstance(destination_container, resource_reference.UnknownResource) and self._has_multiple_top_level_sources
    destination_exists = not isinstance(destination_container, resource_reference.UnknownResource)
    destination_is_existing_dir = destination_exists and destination_container.is_container()
    treat_destination_as_existing_dir = destination_is_existing_dir or (not destination_exists and destination_container.storage_url.url_string.endswith(destination_container.storage_url.delimiter))
    if is_top_level_source_object_name_conflict_possible or (expanded_url_is_valid_parent and treat_destination_as_existing_dir):
        source_delimiter = source.resource.storage_url.delimiter
        relative_path_characters_end_source_prefix = [source_prefix_to_ignore.endswith(source_delimiter + i) for i in _RELATIVE_PATH_SYMBOLS]
        source_url_scheme_string = source.expanded_url.scheme.value + '://'
        source_prefix_to_ignore_without_scheme = source_prefix_to_ignore[len(source_url_scheme_string):]
        source_is_relative_path_symbol = source_prefix_to_ignore_without_scheme in _RELATIVE_PATH_SYMBOLS
        if not any(relative_path_characters_end_source_prefix) and (not source_is_relative_path_symbol):
            source_prefix_to_ignore, _, _ = source_prefix_to_ignore.rpartition(source.expanded_url.delimiter)
        if not source_prefix_to_ignore:
            source_prefix_to_ignore = source.expanded_url.scheme.value + '://'
    full_source_url = source.resource.storage_url.versionless_url_string
    delimiter = source.resource.storage_url.delimiter
    suffix_for_destination = delimiter + full_source_url.split(source_prefix_to_ignore)[1].lstrip(delimiter)
    source_delimiter = source.resource.storage_url.delimiter
    destination_delimiter = destination_container.storage_url.delimiter
    if source_delimiter != destination_delimiter:
        return suffix_for_destination.replace(source_delimiter, destination_delimiter)
    return suffix_for_destination