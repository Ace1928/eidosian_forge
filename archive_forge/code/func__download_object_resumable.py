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
def _download_object_resumable(self, cloud_resource, download_stream, digesters, progress_callback, start_byte, end_byte):
    progress_state = {'start_byte': start_byte, 'end_byte': end_byte}

    def _call_download_object():
        return self._download_object(cloud_resource, download_stream, digesters, progress_callback, progress_state['start_byte'], progress_state['end_byte'])

    def _should_retry_resumable_download(exc_type, exc_value, exc_traceback, state):
        for retryable_error_type in s3transfer.utils.S3_RETRYABLE_DOWNLOAD_ERRORS:
            if isinstance(exc_value, retryable_error_type):
                start_byte = download_stream.tell()
                if start_byte > progress_state['start_byte']:
                    progress_state['start_byte'] = start_byte
                    state.retrial = 0
                log.debug('Retrying download from byte {} after exception: {}. Trace: {}'.format(start_byte, exc_type, exc_traceback))
                return True
        return False
    retryer = retry.Retryer(max_retrials=properties.VALUES.storage.max_retries.GetInt(), wait_ceiling_ms=properties.VALUES.storage.max_retry_delay.GetInt() * 1000, exponential_sleep_multiplier=properties.VALUES.storage.exponential_sleep_multiplier.GetInt())
    return retryer.RetryOnException(_call_download_object, sleep_ms=properties.VALUES.storage.base_retry_delay.GetInt() * 1000, should_retry_if=_should_retry_resumable_download)