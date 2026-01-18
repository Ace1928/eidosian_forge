from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Any
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
import six
def Walk(self, hidden=False, universe_compatible=False, restrict=None):
    """Calls self.Visit() on each node in the CLI tree.

    The walk is DFS, ordered by command name for reproducability.

    Args:
      hidden: Include hidden groups and commands if True.
      universe_compatible: Exclusively include commands which are marked
        universe compatible.
      restrict: Restricts the walk to the command/group dotted paths in this
        list. For example, restrict=['gcloud.alpha.test', 'gcloud.topic']
        restricts the walk to the 'gcloud topic' and 'gcloud alpha test'
        commands/groups. When provided here, parent groups will still be visited
        as the walk progresses down to these leaves, but only parent groups
        between the restrictions and the root.

    Returns:
      The return value of the top level Visit() call.
    """

    def _IsUniverseCompatible(command: Any) -> bool:
        """Determines if a command is universe compatible.

      Args:
        command: CommandCommon command node.

      Returns:
        True if command is universe compatible.
      """
        return not isinstance(command, dict) and command.IsUniverseCompatible()

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

    def _Walk(node, parent):
        """Walk() helper that calls self.Visit() on each node in the CLI tree.

      Args:
        node: CommandCommon tree node.
        parent: The parent Visit() return value, None at the top level.

      Returns:
        The return value of the outer Visit() call.
      """
        if not node.is_group:
            self._Visit(node, parent, is_group=False)
            return parent
        parent = self._Visit(node, parent, is_group=True)
        commands_and_groups = []
        if node.commands:
            for name, command in six.iteritems(node.commands):
                if _Include(command):
                    commands_and_groups.append((name, command, False))
        if node.groups:
            for name, command in six.iteritems(node.groups):
                if _Include(command, traverse=True):
                    commands_and_groups.append((name, command, True))
        for _, command, is_group in sorted(commands_and_groups):
            if is_group:
                _Walk(command, parent)
            else:
                self._Visit(command, parent, is_group=False)
        return parent
    self._num_visited = 0
    parent = None
    for root in self._roots:
        parent = _Walk(root, None)
    self.Done()
    return parent