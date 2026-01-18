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
def _get_write_object_spec(self, size=None):
    """Returns the WriteObjectSpec instance.

    Args:
      size (int|None): Expected object size in bytes.

    Returns:
      (gapic_clients.storage_v2.types.storage.WriteObjectSpec) The
      WriteObjectSpec instance.
    """
    destination_object = self._client.types.Object(name=self._destination_resource.storage_url.object_name, bucket=grpc_util.get_full_bucket_name(self._destination_resource.storage_url.bucket_name), size=size)
    self._set_metadata_if_source_is_object_resource(destination_object)
    metadata_util.update_object_metadata_from_request_config(destination_object, self._request_config, self._source_resource)
    return self._client.types.WriteObjectSpec(resource=destination_object, if_generation_match=copy_util.get_generation_match_value(self._request_config), if_metageneration_match=self._request_config.precondition_metageneration_match, object_size=size)