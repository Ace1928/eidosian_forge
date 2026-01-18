from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.storage import expansion
from googlecloudsdk.command_lib.storage import paths
from googlecloudsdk.command_lib.storage import storage_parallel
from googlecloudsdk.core import exceptions
def _GetExpander(self, path):
    """Get the correct expander for this type of path."""
    if path.is_remote:
        return self._gcs_expander
    return self._local_expander