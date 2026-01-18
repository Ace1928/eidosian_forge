from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import log
Initializes task.

    Args:
      bucket_url (StorageUrl): Launch a bulk restore operation for this bucket.
      object_globs (list[str]): Objects in the target bucket matching these glob
        patterns will be restored.
      allow_overwrite (bool): Overwrite existing live objects.
      deleted_after_time (datetime): Filter results to objects soft-deleted
        after this time. Backend will reject if used with `live_at_time`.
      deleted_before_time (datetime): Filter results to objects soft-deleted
        before this time. Backend will reject if used with `live_at_time`.
      user_request_args (UserRequestArgs|None): Contains restore settings.
    