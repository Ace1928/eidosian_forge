from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import copy
import functools
import os
from googlecloudsdk.api_lib.storage import retry_util as storage_retry_util
from googlecloudsdk.api_lib.storage.gcs_grpc import grpc_util
from googlecloudsdk.api_lib.storage.gcs_grpc import metadata_util
from googlecloudsdk.api_lib.storage.gcs_grpc import retry_util
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import scaled_integer
import six
def _call_write_object(self, first_message):
    """Calls write object api method with routing header.

    Args:
      first_message (WriteObjectSpec|str): WriteObjectSpec for Simple uploads.
    Returns:
      (gapic_clients.storage_v2.types.WriteObjectResponse) Request response.
    """
    return self._client.storage.write_object(requests=self._upload_write_object_request_generator(first_message=first_message), metadata=metadata_util.get_bucket_name_routing_header(grpc_util.get_full_bucket_name(self._destination_resource.storage_url.bucket_name)))