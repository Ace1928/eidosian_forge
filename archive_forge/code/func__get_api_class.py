from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.api_lib.storage.gcs_grpc import client as gcs_grpc_client
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.storage.gcs_xml import client as gcs_xml_client
from googlecloudsdk.api_lib.storage.s3_xml import client as s3_xml_client
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _get_api_class(provider):
    """Returns a CloudApi subclass corresponding to the requested provider.

  Args:
    provider (storage_url.ProviderPrefix): Cloud provider prefix.

  Returns:
    An appropriate CloudApi subclass.

  Raises:
    Error: If provider is not a cloud scheme in storage_url.ProviderPrefix.
  """
    if provider == storage_url.ProviderPrefix.GCS:
        if properties.VALUES.storage.use_grpc_if_available.GetBool() or properties.VALUES.storage.preferred_api.Get() == properties.StoragePreferredApi.GRPC_WITH_JSON_FALLBACK.value:
            log.debug('Using gRPC client with JSON Fallback.')
            return gcs_grpc_client.GrpcClientWithJsonFallback
        if properties.VALUES.storage.gs_xml_access_key_id.Get() and properties.VALUES.storage.gs_xml_secret_access_key.Get():
            return gcs_xml_client.XmlClient
        return gcs_json_client.JsonClient
    elif provider == storage_url.ProviderPrefix.S3:
        return s3_xml_client.S3XmlClient
    else:
        raise errors.Error(_INVALID_PROVIDER_PREFIX_MESSAGE)