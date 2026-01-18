from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
import boto3
import botocore
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage import headers_util
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.api_lib.storage import xml_metadata_field_converters
from googlecloudsdk.api_lib.storage import xml_metadata_util
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.resources import s3_resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import download_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import scaled_integer
import s3transfer
def _upload_using_managed_transfer_utility(self, source_stream, destination_resource, extra_args):
    """Uploads the data using boto3's managed transfer utility.

    Calls the upload_fileobj method which performs multi-threaded multipart
    upload automatically. Performs slightly better than put_object API method.
    However, upload_fileobj cannot perform data intergrity checks and we have
    to use put_object method in such cases.

    Args:
      source_stream (a file-like object): A file-like object to upload. At a
        minimum, it must implement the read method, and must return bytes.
      destination_resource (resource_reference.ObjectResource|UnknownResource):
        Represents the metadata for the destination object.
      extra_args (dict): Extra arguments that may be passed to the client
        operation.

    Returns:
      resource_reference.ObjectResource with uploaded object's metadata.
    """
    bucket_name = destination_resource.storage_url.bucket_name
    object_name = destination_resource.storage_url.object_name
    multipart_chunksize = scaled_integer.ParseInteger(properties.VALUES.storage.multipart_chunksize.Get())
    multipart_threshold = scaled_integer.ParseInteger(properties.VALUES.storage.multipart_threshold.Get())
    self.client.upload_fileobj(Fileobj=source_stream, Config=boto3.s3.transfer.TransferConfig(use_threads=False, multipart_chunksize=multipart_chunksize, multipart_threshold=multipart_threshold), Bucket=bucket_name, Key=object_name, ExtraArgs=extra_args)
    return self.get_object_metadata(bucket_name, object_name, request_config_factory.get_request_config(storage_url.CloudUrl(scheme=storage_url.ProviderPrefix.S3)))