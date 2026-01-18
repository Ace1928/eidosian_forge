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
def _initialize_upload(self):
    """Sets up the upload session and returns the upload id.

    Additionally, it does the following tasks:
    1. Grabs the persisted size on the backend.
    2. Sets the appropiate write offset.
    3. Calls the tracker callback.

    Returns:
      The upload session ID.
    """
    if self._serialization_data is not None:
        upload_id = self._serialization_data['upload_id']
        write_offset = self._get_write_offset(upload_id)
        self._start_offset = write_offset
        log.debug('Write offset after resuming: %s', write_offset)
    else:
        upload_id = super(ResumableUpload, self)._initialize_upload()
    if self._tracker_callback is not None:
        self._tracker_callback({'upload_id': upload_id})
    return upload_id