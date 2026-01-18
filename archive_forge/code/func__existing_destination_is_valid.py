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
def _existing_destination_is_valid(self, destination_resource):
    """Returns True if a completed temporary component can be reused."""
    digesters = upload_util.get_digesters(self._source_resource, destination_resource)
    with upload_util.get_stream(self._transformed_source_resource, length=self._length, offset=self._offset, digesters=digesters) as stream:
        stream.seek(0, whence=os.SEEK_END)
    try:
        upload_util.validate_uploaded_object(digesters, destination_resource, task_status_queue=None)
        return True
    except command_errors.HashMismatchError:
        return False