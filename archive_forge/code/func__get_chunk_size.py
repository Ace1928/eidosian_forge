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
def _get_chunk_size(self):
    """Returns the chunk size corrected to be multiple of MAX_WRITE_CHUNK_BYTES.

    It also sets the attribute _should_log_message if it is needed to log
    the warning message.

    Look at the docstring on StreamingUpload class.

    Returns:
      (int) The chunksize value corrected.
    """
    initial_chunk_size = scaled_integer.ParseInteger(properties.VALUES.storage.upload_chunk_size.Get())
    if initial_chunk_size >= self._client.types.ServiceConstants.Values.MAX_WRITE_CHUNK_BYTES:
        adjust_chunk_size = initial_chunk_size % self._client.types.ServiceConstants.Values.MAX_WRITE_CHUNK_BYTES
        if adjust_chunk_size > 0:
            self._log_chunk_warning = True
        return initial_chunk_size - adjust_chunk_size
    self._log_chunk_warning = True
    return self._client.types.ServiceConstants.Values.MAX_WRITE_CHUNK_BYTES