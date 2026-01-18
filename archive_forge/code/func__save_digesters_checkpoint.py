from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.core.updater import installers
def _save_digesters_checkpoint(self):
    """Updates checkpoint that holds old hashes to optimize backwards seeks."""
    if not self._digesters:
        return
    self._checkpoint_absolute_index = self._get_absolute_position()
    self._checkpoint_digesters = hash_util.copy_digesters(self._digesters)