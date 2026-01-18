from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
def _get_nested_objects_iterator(self, parent_name_expansion_result):
    new_storage_url = parent_name_expansion_result.resource.storage_url.join('**')
    child_resources = self._get_wildcard_iterator(new_storage_url.url_string, managed_folder_setting=self._managed_folder_setting)
    for child_resource in child_resources:
        yield self._get_name_expansion_result(child_resource, parent_name_expansion_result.resource.storage_url, parent_name_expansion_result.original_url)