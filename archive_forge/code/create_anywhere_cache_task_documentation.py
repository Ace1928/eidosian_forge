from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import log
Initializes task.

    Args:
      bucket_url (CloudUrl): The URL of the bucket where the Anywhere Cache
        should be created.
      zone (str): Name of the zonal locations where the Anywhere Cache should be
        created.
      admission_policy (str|None): The cache admission policy decides for each
        cache miss, that is whether to insert the missed block or not.
      ttl (str|None): Cache entry time-to-live in seconds
    