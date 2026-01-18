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
def _print_summary_for_top_level_url(self, resource_url, only_display_buckets, object_count, total_bytes):
    if self._display_detail in (list_util.DisplayDetail.LONG, list_util.DisplayDetail.FULL) and (not only_display_buckets):
        print('TOTAL: {} objects, {} bytes ({})'.format(object_count, int(total_bytes), shim_format_util.get_human_readable_byte_value(total_bytes, self._use_gsutil_style)))