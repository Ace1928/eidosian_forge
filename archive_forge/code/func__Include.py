from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Any
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
import six
def _Include(command, traverse=False):
    """Determines if command should be included in the walk.

      Args:
        command: CommandCommon command node.
        traverse: If True then check traversal through group to subcommands.

      Returns:
        True if command should be included in the walk.
      """
    if not hidden and command.IsHidden():
        return False
    if universe_compatible and (not _IsUniverseCompatible(command)):
        return False
    if not restrict:
        return True
    path = '.'.join(command.GetPath())
    for item in restrict:
        if path.startswith(item):
            return True
        if traverse and item.startswith(path):
            return True
    return False