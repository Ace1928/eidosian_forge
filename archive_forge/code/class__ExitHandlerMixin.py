from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
class _ExitHandlerMixin:
    """Provides an exit handler for copy tasks."""

    def exit_handler(self, error=None, task_status_queue=None):
        """Send copy result info to manifest if requested."""
        if error and self._send_manifest_messages:
            if not task_status_queue:
                raise command_errors.Error('Unable to send message to manifest for source: {}'.format(self._source_resource))
            manifest_util.send_error_message(task_status_queue, self._source_resource, self._destination_resource, error)