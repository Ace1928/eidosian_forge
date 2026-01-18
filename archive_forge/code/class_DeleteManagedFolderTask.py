from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import log
class DeleteManagedFolderTask(CloudDeleteTask):
    """Task to delete a managed folder."""

    @property
    def managed_folder_url(self):
        """The URL of the resource deleted by this task.

    Exposing this allows execution to respect containment order.
    """
        return self._url

    def _make_delete_api_call(self, client, request_config):
        del request_config
        client.delete_managed_folder(self._url.bucket_name, self._url.object_name)