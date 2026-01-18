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
def _catch_client_error_raise_s3_api_error(format_str=None):
    """Decorator that catches botocore ClientErrors and raises XmlApiErrors.

  Args:
    format_str (str): A googlecloudsdk.api_lib.storage.errors.S3ErrorPayload
      format string. Note that any properties that are accessed here are on the
      S3ErrorPayload object, not the object returned from botocore.

  Returns:
    A decorator that catches botocore.exceptions.ClientError and returns an
      XmlApiError with a formatted error message.
  """
    return errors.catch_error_raise_cloud_api_error([(botocore.exceptions.ClientError, None, errors.XmlApiError)], format_str=format_str)