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
def _expand_destination_wildcards(destination_string):
    """Expands destination wildcards.

  Ensures that only one resource matches the wildcard expanded string. Much
  like the unix cp command, the storage surface only supports copy operations
  to one user-specified destination.

  Args:
    destination_string (str): A string representing the destination url.

  Returns:
    A resource_reference.Resource, or None if no matching resource is found.

  Raises:
    InvalidUrlError if more than one resource is matched, or the source
      contained an unescaped wildcard and no resources were matched.
  """
    destination_iterator = plurality_checkable_iterator.PluralityCheckableIterator(wildcard_iterator.get_wildcard_iterator(destination_string, fields_scope=cloud_api.FieldsScope.SHORT))
    if destination_iterator.is_plural():
        raise errors.InvalidUrlError('Destination ({}) must match exactly one URL.'.format(destination_string))
    contains_unexpanded_wildcard = destination_iterator.is_empty() and wildcard_iterator.contains_wildcard(destination_string)
    if contains_unexpanded_wildcard:
        raise errors.InvalidUrlError('Destination ({}) contains an unexpected wildcard.'.format(destination_string))
    if not destination_iterator.is_empty():
        return next(destination_iterator)