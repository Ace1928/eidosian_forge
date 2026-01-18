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
class SimpleUpload(_Upload):
    """Uploads an object with a single request."""

    @retry_util.grpc_default_retryer
    def run(self):
        """Uploads the object in non-resumable mode.

    Returns:
      (gapic_clients.storage_v2.types.WriteObjectResponse) A WriteObjectResponse
      instance.
    """
        write_object_spec = self._get_write_object_spec(self._request_config.resource_args.size)
        return self._call_write_object(write_object_spec)