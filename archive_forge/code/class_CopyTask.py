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
class CopyTask(task.Task):
    """Parent task that handles common attributes and an __init__ status print."""

    def __init__(self, source_resource, destination_resource, print_created_message=False, print_source_version=False, user_request_args=None, verbose=False):
        """Initializes task.

    Args:
      source_resource (resource_reference.Resource): Source resource to copy.
      destination_resource (resource_reference.Resource): Target resource to
        copy to.
      print_created_message (bool): Print a message containing the URL of the
        copy result.
      print_source_version (bool): Print source object version in status message
        enabled by the `verbose` kwarg.
      user_request_args (UserRequestArgs|None): Various user-set values
        typically converted to an API-specific RequestConfig.
      verbose (bool): Print a "copying" status message on initialization.
    """
        super(CopyTask, self).__init__()
        self._source_resource = source_resource
        self._destination_resource = destination_resource
        self._print_created_message = print_created_message
        self._print_source_version = print_source_version
        self._user_request_args = user_request_args
        self._verbose = verbose
        self._send_manifest_messages = bool(self._user_request_args and self._user_request_args.manifest_path)
        if verbose:
            if self._print_source_version:
                source_string = source_resource.storage_url.url_string
            else:
                source_string = source_resource.storage_url.versionless_url_string
            log.status.Print('Copying {} to {}'.format(source_string, destination_resource.storage_url.versionless_url_string))

    def _print_created_message_if_requested(self, resource):
        if self._print_created_message:
            log.status.Print('Created: {}'.format(resource))