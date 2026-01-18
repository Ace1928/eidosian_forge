from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def _get_parent_string_from_bucket(bucket):
    gcs_client = gcs_json_client.JsonClient()
    bucket_resource = gcs_client.get_bucket(bucket)
    return _get_parent_string(bucket_resource.metadata.projectNumber, bucket_resource.metadata.location.lower())