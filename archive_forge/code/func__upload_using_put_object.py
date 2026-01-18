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
def _upload_using_put_object(self, source_stream, destination_resource, extra_args):
    """Uploads the source stream using the put_object API method.

    Args:
      source_stream (a seekable file-like object): The stream of bytes to be
        uploaded.
      destination_resource (resource_reference.ObjectResource|UnknownResource):
        Represents the metadata for the destination object.
      extra_args (dict): Extra arguments that may be passed to the client
        operation.

    Returns:
      resource_reference.ObjectResource with uploaded object's metadata.
    """
    kwargs = {'Bucket': destination_resource.storage_url.bucket_name, 'Key': destination_resource.storage_url.object_name, 'Body': source_stream}
    kwargs.update(extra_args)
    response = self.client.put_object(**kwargs)
    return xml_metadata_util.get_object_resource_from_xml_response(self.scheme, response, destination_resource.storage_url.bucket_name, destination_resource.storage_url.object_name)