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
def _get_write_offset(self, upload_id):
    """Returns the amount of data persisted on the server.

    Args:
      upload_id (str): Session URI for resumable upload operation.
    Returns:
      (int) The total number of bytes that have been persisted for an object
      on the server. This value can be used as the write_offset.
    """
    request = self._client.types.QueryWriteStatusRequest(upload_id=upload_id)
    return self._client.storage.query_write_status(request=request).persisted_size