from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import functools
import os
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks.cp import file_part_task
from googlecloudsdk.command_lib.storage.tasks.cp import upload_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
def _handle_resumable_upload_error(exc_type, exc_value, exc_traceback, state):
    """Returns true if resumable upload should retry on error argument."""
    del exc_traceback
    if not (exc_type is api_errors.NotFoundError or getattr(exc_value, 'status_code', None) == 410):
        if exc_type is api_errors.ResumableUploadAbortError:
            tracker_file_util.delete_tracker_file(tracker_file_path)
        return False
    tracker_file_util.delete_tracker_file(tracker_file_path)
    if state.retrial == 0:
        try:
            api.get_bucket(self._destination_resource.storage_url.bucket_name)
        except api_errors.CloudApiError as e:
            status = getattr(e, 'status_code', None)
            if status not in (401, 403):
                raise
    return True