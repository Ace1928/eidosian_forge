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
class RecoverableUpload(_Upload):
    """Common logic for strategies allowing retries in-flight."""

    def _initialize_upload(self):
        """Sets up the upload session and returns the upload id.

    This method sets the start offset to 0.

    Returns:
      (str) Session URI for resumable upload operation.
    """
        write_object_spec = self._get_write_object_spec()
        request = self._client.types.StartResumableWriteRequest(write_object_spec=write_object_spec)
        upload_id = self._client.storage.start_resumable_write(request=request).upload_id
        self._start_offset = 0
        return upload_id

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

    def _should_retry(self, upload_id, exc_type=None, exc_value=None, exc_traceback=None, state=None):
        if not retry_util.is_retriable(exc_type, exc_value, exc_traceback, state):
            return False
        persisted_size = self._get_write_offset(upload_id)
        is_progress_made_since_last_uplaod = persisted_size > self._start_offset
        if is_progress_made_since_last_uplaod:
            self._start_offset = persisted_size
        return True

    def _perform_upload(self, upload_id):
        return self._call_write_object(upload_id)

    def run(self):
        upload_id = self._initialize_upload()
        new_should_retry = functools.partial(self._should_retry, upload_id)
        return storage_retry_util.retryer(target=self._perform_upload, should_retry_if=new_should_retry, target_args=[upload_id])