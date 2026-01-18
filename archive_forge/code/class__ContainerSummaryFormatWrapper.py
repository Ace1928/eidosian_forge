from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import list_util
from googlecloudsdk.command_lib.storage.resources import shim_format_util
class _ContainerSummaryFormatWrapper(list_util.BaseFormatWrapper):
    """For formatting how containers are printed when listed by du."""

    def __init__(self, resource, container_size=None, object_state=None, readable_sizes=False, use_gsutil_style=False):
        """See list_util.BaseFormatWrapper class for function doc strings."""
        super(_ContainerSummaryFormatWrapper, self).__init__(resource, display_detail=list_util.DisplayDetail.SHORT, object_state=object_state, readable_sizes=readable_sizes, use_gsutil_style=use_gsutil_style)
        self._container_size = container_size

    def __str__(self):
        """Returns string of select properties from resource."""
        raw_url_string = self.resource.storage_url.versionless_url_string
        if self.resource.storage_url.is_bucket():
            url_string = raw_url_string.rstrip('/')
        else:
            url_string = raw_url_string
        size = list_util.check_and_convert_to_readable_sizes(self._container_size, self._readable_sizes, self._use_gsutil_style)
        return '{size:<13}{url}'.format(size=size, url=url_string)