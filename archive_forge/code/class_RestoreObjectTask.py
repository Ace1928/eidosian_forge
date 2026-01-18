from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import log
class RestoreObjectTask(task.Task):
    """Restores a soft-deleted cloud storage object."""

    def __init__(self, object_resource, user_request_args=None):
        """Initializes task.

    Args:
      object_resource (resource_reference.ObjectResource): Object to restore.
      user_request_args (UserRequestArgs|None): Contains restore settings.
    """
        super(RestoreObjectTask, self).__init__()
        self._object_resource = object_resource
        self._user_request_args = user_request_args

    def execute(self, task_status_queue=None):
        log.status.Print('Restoring {}...'.format(self._object_resource))
        provider = self._object_resource.storage_url.scheme
        request_config = request_config_factory.get_request_config(self._object_resource.storage_url, user_request_args=self._user_request_args)
        api_factory.get_api(provider).restore_object(self._object_resource.storage_url, request_config=request_config)
        if task_status_queue:
            progress_callbacks.increment_count_callback(task_status_queue)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._object_resource == other._object_resource and self._user_request_args == other._user_request_args