from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import walker
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projector
def _Prune(self, command):
    """Returns True if command should be pruned from the CLI tree.

    Branch pruning is mainly for generating static unit test data. The static
    tree for the entire CLI would be an unnecessary burden on the depot.

    self._branch, if not None, is already split into a path with the first
    name popped. If branch is not a prefix of command.GetPath()[1:] it will
    be pruned.

    Args:
      command: The calliope Command object to check.

    Returns:
      True if command should be pruned from the CLI tree.
    """
    if not self._branch:
        return False
    path = command.GetPath()
    if len(path) < 2:
        return False
    path = path[1:]
    if path[0] in ('alpha', 'beta'):
        path = path[1:]
    for name in self._branch:
        if not path:
            return False
        if path[0] != name:
            return True
        path.pop(0)
    return False