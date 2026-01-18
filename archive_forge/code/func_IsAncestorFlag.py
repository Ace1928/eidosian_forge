from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import walker
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projector
def IsAncestorFlag(self, flag):
    """Determines if flag is provided by an ancestor command.

    NOTE: This function is used to allow for global flags to be added in at the
          top level but not in subgroups/commands
    Args:
      flag: str, The flag name (no leading '-').

    Returns:
      bool, True if flag provided by an ancestor command, false if not.
    """
    command = self._parent
    while command:
        if flag in command:
            return True
        command = command._parent
    return False