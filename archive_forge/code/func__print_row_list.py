from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import shim_format_util
import six
def _print_row_list(self, resource_wrappers, resource_url, only_display_buckets):
    """Prints ResourceWrapper objects in list with custom row formatting."""
    object_count = total_bytes = 0
    terminator = '\x00' if self._zero_terminator else '\n'
    for i, resource_wrapper in enumerate(resource_wrappers):
        resource_wrapper_string = six.text_type(resource_wrapper)
        if isinstance(resource_wrapper.resource, resource_reference.ObjectResource):
            object_count += 1
            total_bytes += resource_wrapper.resource.size or 0
        if not resource_wrapper_string:
            continue
        if i == 0 and resource_wrapper and (resource_wrapper_string[0] == '\n'):
            print(resource_wrapper_string[1:], end=terminator)
        else:
            print(resource_wrapper_string, end=terminator)
    self._print_summary_for_top_level_url(resource_url=resource_url, only_display_buckets=only_display_buckets, object_count=object_count, total_bytes=total_bytes)
    return total_bytes