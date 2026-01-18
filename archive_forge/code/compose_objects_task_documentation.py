from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import log
Initializes task.

    Args:
      source_resources (list[ObjectResource|UnknownResource]): The objects to
        compose. This field accepts UnknownResources since it should allow
        ComposeObjectsTasks to be initialized before the target objects have
        been created.
      destination_resource (resource_reference.UnknownResource): Metadata for
        the resulting composite object.
      original_source_resource (Resource|None): Useful for finding metadata to
        apply to final object. For instance, if doing a composite upload, this
        would represent the pre-split local file.
      posix_to_set (PosixAttributes|None): POSIX info set as custom cloud
        metadata on target. If preserving POSIX, avoids re-parsing metadata from
        file system.
      print_status_message (bool): If True, the task prints the status message.
      user_request_args (UserRequestArgs|None): Values for RequestConfig.
    