from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import copy
import json
from apitools.base.py import encoding_helper
from apitools.base.py import transfer
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage import retry_util
from googlecloudsdk.api_lib.storage.gcs_json import metadata_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import scaled_integer
import six
class _BaseRecoverableUpload(_Upload):
    """Common logic for strategies allowing retries in-flight."""

    def _get_upload(self):
        """Returns an apitools upload class used for a new transfer."""
        resource_args = self._request_config.resource_args
        size = getattr(resource_args, 'size', None)
        max_retries = properties.VALUES.storage.max_retries.GetInt()
        apitools_upload = transfer.Upload(self._source_stream, resource_args.content_type, auto_transfer=False, chunksize=scaled_integer.ParseInteger(properties.VALUES.storage.upload_chunk_size.Get()), gzip_encoded=self._should_gzip_in_flight, total_size=size, num_retries=max_retries)
        apitools_upload.strategy = transfer.RESUMABLE_UPLOAD
        return apitools_upload

    def _initialize_upload(self):
        """Inserts a a new object at the upload destination."""
        if not self._apitools_upload.initialized:
            self._gcs_api.client.objects.Insert(self._get_validated_insert_request(), upload=self._apitools_upload)

    @abc.abstractmethod
    def _call_appropriate_apitools_upload_strategy(self):
        """Responsible for pushing bytes to GCS with an appropriate strategy."""
        pass

    def _should_retry_resumable_upload(self, exc_type, exc_value, exc_traceback, state):
        """Returns True if the failure should be retried."""
        if not isinstance(exc_value, errors.RetryableApiError):
            return False
        self._apitools_upload.RefreshResumableUploadState()
        if self._apitools_upload.progress > self._last_progress_byte:
            self._last_progress_byte = self._apitools_upload.progress
            state.retrial = 0
        log.debug('Retrying upload after exception: {}. Trace: {}'.format(exc_type, exc_traceback))
        return True

    def run(self):
        """Uploads with in-flight retry logic and returns an Object message."""
        self._apitools_upload = self._get_upload()
        self._apitools_upload.bytes_http = self._http_client
        retry_util.set_retry_func(self._apitools_upload)
        self._initialize_upload()
        self._last_progress_byte = self._apitools_upload.progress
        try:
            http_response = retry_util.retryer(target=self._call_appropriate_apitools_upload_strategy, should_retry_if=self._should_retry_resumable_upload)
        except retry.MaxRetrialsException as e:
            raise errors.ResumableUploadAbortError('Max retrial attempts reached. Aborting upload.Error: {}'.format(e))
        return self._gcs_api.client.objects.ProcessHttpResponse(self._gcs_api.client.objects.GetMethodConfig('Insert'), http_response)