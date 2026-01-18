from collections import deque
from . import errors, revision
def _find_tails(self):
    return [node for node in self._nodes.values() if not node.parent_keys]