from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.images import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
@staticmethod
def _GetDaisyBucket(args):
    storage_client = storage_api.StorageClient()
    bucket_location = storage_client.GetBucketLocationForFile(args.destination_uri)
    return daisy_utils.CreateDaisyBucketInProject(bucket_location, storage_client, enable_uniform_level_access=True)