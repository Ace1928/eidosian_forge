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
def _list_url(self, raw_cloud_url):
    """Recursively create wildcard iterators to print all relevant items."""
    fields_scope = _translate_display_detail_to_fields_scope(self._display_detail, is_bucket_listing=raw_cloud_url.is_provider())
    if self._include_managed_folders:
        managed_folder_setting = folder_util.ManagedFolderSetting.LIST_AS_PREFIXES
    else:
        managed_folder_setting = folder_util.ManagedFolderSetting.DO_NOT_LIST
    resources = plurality_checkable_iterator.PluralityCheckableIterator(wildcard_iterator.CloudWildcardIterator(raw_cloud_url, error_on_missing_key=False, exclude_patterns=self._exclude_patterns, fetch_encrypted_object_hashes=self._fetch_encrypted_object_hashes, fields_scope=fields_scope, get_bucket_metadata=self._buckets_flag, halt_on_empty_response=self._halt_on_empty_response, managed_folder_setting=managed_folder_setting, next_page_token=self._next_page_token, object_state=self._object_state))
    if resources.is_empty():
        raise errors.InvalidUrlError('One or more URLs matched no objects.')
    only_display_buckets = self._should_only_display_buckets(raw_cloud_url)
    if only_display_buckets:
        resources_wrappers = self._recursion_helper(resources, recursion_level=0)
    elif self._recursion_flag and '**' not in raw_cloud_url.url_string:
        print_top_level_container = True
        if raw_cloud_url.is_bucket():
            print_top_level_container = False
        resources_wrappers = self._recursion_helper(resources, float('inf'), print_top_level_container)
    elif not resources.is_plural() and resource_reference.is_container_or_has_container_url(resources.peek()):
        resources_wrappers = self._get_container_iterator(resources.peek().storage_url, recursion_level=0)
    else:
        resources_wrappers = self._recursion_helper(resources, recursion_level=1)
    size_in_bytes = 0
    if self._display_detail == DisplayDetail.JSON:
        self._print_json_list(resources_wrappers)
    else:
        size_in_bytes = self._print_row_list(resources_wrappers, raw_cloud_url, only_display_buckets)
    return size_in_bytes