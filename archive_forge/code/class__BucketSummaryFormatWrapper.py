from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import list_util
from googlecloudsdk.command_lib.storage.resources import shim_format_util
class _BucketSummaryFormatWrapper(_ContainerSummaryFormatWrapper):

    def __str__(self):
        if self.resource.storage_url.is_bucket():
            return super(_BucketSummaryFormatWrapper, self).__str__()
        else:
            return ''