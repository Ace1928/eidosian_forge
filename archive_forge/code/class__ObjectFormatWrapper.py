from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import list_util
from googlecloudsdk.command_lib.storage.resources import shim_format_util
class _ObjectFormatWrapper(list_util.BaseFormatWrapper):
    """For formatting how obejects are printed when listed by du."""

    def __str__(self):
        """Returns string of select properties from resource."""
        size = getattr(self.resource, 'size', 0)
        url_string, _ = self._check_and_handles_versions()
        return '{size:<13}{url}'.format(size=list_util.check_and_convert_to_readable_sizes(size, self._readable_sizes, use_gsutil_style=self._use_gsutil_style), url=url_string)