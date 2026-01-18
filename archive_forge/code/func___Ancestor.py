from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import walker
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projector
def __Ancestor(self, flag):
    """Determines if flag is provided by an ancestor command.

    Args:
      flag: str, The flag name (no leading '-').

    Returns:
      bool, True if flag provided by an ancestor command, false if not.
    """
    command = self._parent
    while command:
        if flag in command.flags:
            return True
        command = command._parent
    return False