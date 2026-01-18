import collections
import io
import itertools
import os
from taskflow.types import graph
from taskflow.utils import iter_utils
from taskflow.utils import misc
class _BFSIter(object):
    """Breadth first iterator (non-recursive) over the child nodes."""

    def __init__(self, root, include_self=False, right_to_left=False):
        self.root = root
        self.right_to_left = bool(right_to_left)
        self.include_self = bool(include_self)

    def __iter__(self):
        q = collections.deque()
        if self.include_self:
            q.append(self.root)
        elif self.right_to_left:
            q.extend(iter(self.root))
        else:
            q.extend(self.root.reverse_iter())
        while q:
            node = q.popleft()
            yield node
            if self.right_to_left:
                q.extend(iter(node))
            else:
                q.extend(node.reverse_iter())