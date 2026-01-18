from collections import deque
from . import errors, revision
class _KnownGraphNode:
    """Represents a single object in the known graph."""
    __slots__ = ('key', 'parent_keys', 'child_keys', 'gdfo')

    def __init__(self, key, parent_keys):
        self.key = key
        self.parent_keys = parent_keys
        self.child_keys = []
        self.gdfo = None

    def __repr__(self):
        return '{}({}  gdfo:{} par:{} child:{})'.format(self.__class__.__name__, self.key, self.gdfo, self.parent_keys, self.child_keys)