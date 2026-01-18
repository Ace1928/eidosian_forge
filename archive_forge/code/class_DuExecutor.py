from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import list_util
from googlecloudsdk.command_lib.storage.resources import shim_format_util
class DuExecutor(list_util.BaseListExecutor):
    """Helper class for the Du command."""

    def __init__(self, cloud_urls, exclude_patterns=None, object_state=None, readable_sizes=False, summarize=False, total=False, use_gsutil_style=False, zero_terminator=False):
        """See list_util.BaseListExecutor class for function doc strings."""
        super(DuExecutor, self).__init__(cloud_urls=cloud_urls, exclude_patterns=exclude_patterns, object_state=object_state, readable_sizes=readable_sizes, recursion_flag=True, total=total, use_gsutil_style=use_gsutil_style, zero_terminator=zero_terminator)
        self._summarize = summarize
        if self._summarize:
            self._container_summary_wrapper = _BucketSummaryFormatWrapper
        else:
            self._container_summary_wrapper = _ContainerSummaryFormatWrapper
            self._object_wrapper = _ObjectFormatWrapper

    def _should_only_display_buckets(self, raw_cloud_url):
        return False

    def _print_summary_for_top_level_url(self, resource_url, only_display_buckets, object_count, total_bytes):
        if not self._summarize or resource_url.is_provider():
            return
        if self._readable_sizes:
            total_bytes = shim_format_util.get_human_readable_byte_value(total_bytes, use_gsutil_style=self._use_gsutil_style)
        if resource_url.is_bucket():
            url_string = resource_url.url_string.rstrip('/')
        else:
            url_string = resource_url.url_string
        print('{size:<13}{url}'.format(size=total_bytes, url=url_string), end='\x00' if self._zero_terminator else '\n')

    def _print_total(self, all_sources_total_bytes):
        print('{size:<13}total'.format(size=list_util.check_and_convert_to_readable_sizes(all_sources_total_bytes, self._readable_sizes, self._use_gsutil_style)), end='\x00' if self._zero_terminator else '\n')