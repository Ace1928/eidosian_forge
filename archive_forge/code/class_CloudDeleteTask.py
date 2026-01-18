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
class CloudDeleteTask(DeleteTask):
    """Base class for tasks that delete a cloud resource."""

    @abc.abstractmethod
    def _make_delete_api_call(self, client, request_config):
        """Performs an API call to delete a resource. Overridden by children."""
        raise NotImplementedError

    def _perform_deletion(self):
        client = api_factory.get_api(self._url.scheme)
        request_config = request_config_factory.get_request_config(self._url, user_request_args=self._user_request_args)
        return self._make_delete_api_call(client, request_config)