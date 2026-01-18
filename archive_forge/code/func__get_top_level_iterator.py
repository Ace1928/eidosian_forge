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
def _get_top_level_iterator(self):
    """Iterates over user-entered URLs and does initial processing."""
    for url in self._urls_iterator:
        original_storage_url = storage_url.storage_url_from_string(url)
        if isinstance(original_storage_url, storage_url.CloudUrl) and original_storage_url.is_bucket() and (self._recursion_requested is not RecursionSetting.YES) and (self._include_buckets is BucketSetting.NO_WITH_ERROR):
            raise errors.InvalidUrlError('Expected object URL. Received: {}'.format(url))
        self._url_found_match_tracker[url] = self._url_found_match_tracker.get(url, False)
        if self._managed_folder_setting in {folder_util.ManagedFolderSetting.LIST_WITH_OBJECTS, folder_util.ManagedFolderSetting.LIST_WITHOUT_OBJECTS} and self._recursion_requested is RecursionSetting.YES:
            wildcard_iterator_managed_folder_setting = folder_util.ManagedFolderSetting.LIST_AS_PREFIXES
        else:
            wildcard_iterator_managed_folder_setting = self._managed_folder_setting
        for resource in self._get_wildcard_iterator(url, managed_folder_setting=wildcard_iterator_managed_folder_setting):
            if self._managed_folder_setting is folder_util.ManagedFolderSetting.LIST_WITHOUT_OBJECTS and isinstance(resource, resource_reference.ObjectResource):
                continue
            yield (url, self._get_name_expansion_result(resource, resource.storage_url, original_storage_url))