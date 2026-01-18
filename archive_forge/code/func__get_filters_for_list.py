from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def _get_filters_for_list(self, source_bucket, destination):
    """Returns the filter string used for list API call."""
    filter_list = []
    if source_bucket is not None:
        filter_list.append('objectMetadataReportOptions.storageFilters.bucket="{}"'.format(source_bucket.bucket_name))
    if destination is not None:
        filter_list.append('objectMetadataReportOptions.storageDestinationOptions.bucket="{}"'.format(destination.bucket_name))
        if destination.object_name is not None:
            filter_list.append('objectMetadataReportOptions.storageDestinationOptions.destinationPath="{}"'.format(destination.object_name))
    if filter_list:
        return ' AND '.join(filter_list)
    else:
        return None