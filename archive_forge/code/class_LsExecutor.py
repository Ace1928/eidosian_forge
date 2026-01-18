from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import list_util
from googlecloudsdk.command_lib.storage.resources import gcloud_full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import gsutil_full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.resources import shim_format_util
class LsExecutor(list_util.BaseListExecutor):
    """Helper class for the ls command."""

    def __init__(self, cloud_urls, buckets_flag=False, display_detail=list_util.DisplayDetail.SHORT, fetch_encrypted_object_hashes=False, halt_on_empty_response=True, include_etag=False, include_managed_folders=False, next_page_token=None, object_state=None, readable_sizes=False, recursion_flag=False, use_gsutil_style=False):
        """See list_util.BaseListExecutor class for function doc strings."""
        super(LsExecutor, self).__init__(cloud_urls=cloud_urls, buckets_flag=buckets_flag, display_detail=display_detail, fetch_encrypted_object_hashes=fetch_encrypted_object_hashes, halt_on_empty_response=halt_on_empty_response, include_etag=include_etag, include_managed_folders=include_managed_folders, next_page_token=next_page_token, object_state=object_state, readable_sizes=readable_sizes, recursion_flag=recursion_flag, use_gsutil_style=use_gsutil_style)
        if use_gsutil_style:
            self._full_formatter = gsutil_full_resource_formatter.GsutilFullResourceFormatter()
        else:
            self._full_formatter = gcloud_full_resource_formatter.GcloudFullResourceFormatter()
        self._header_wrapper = _HeaderFormatWrapper
        self._object_wrapper = _ResourceFormatWrapper

    def _print_summary_for_top_level_url(self, resource_url, only_display_buckets, object_count, total_bytes):
        if self._display_detail in (list_util.DisplayDetail.LONG, list_util.DisplayDetail.FULL) and (not only_display_buckets):
            print('TOTAL: {} objects, {} bytes ({})'.format(object_count, int(total_bytes), shim_format_util.get_human_readable_byte_value(total_bytes, self._use_gsutil_style)))

    def _print_json_list(self, resource_wrappers):
        """Prints ResourceWrapper objects as JSON list."""
        is_empty_list = True
        for i, resource_wrapper in enumerate(resource_wrappers):
            is_empty_list = False
            if i == 0:
                print('[')
                print(resource_wrapper, end='')
            else:
                print(',\n{}'.format(resource_wrapper), end='')
        print()
        if not is_empty_list:
            print(']')