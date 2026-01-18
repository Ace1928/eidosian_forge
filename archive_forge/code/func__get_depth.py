import os
import stat
import sys
import warnings
from contextlib import suppress
from io import BytesIO
from typing import (
from .errors import NotTreeError
from .file import GitFile
from .objects import (
from .pack import (
from .protocol import DEPTH_INFINITE
from .refs import PEELED_TAG_SUFFIX, Ref
def _get_depth(self, head, get_parents=lambda commit: commit.parents, max_depth=None):
    """Return the current available depth for the given head.
        For commits with multiple parents, the largest possible depth will be
        returned.

        Args:
            head: commit to start from
            get_parents: optional function for getting the parents of a commit
            max_depth: maximum depth to search
        """
    if head not in self:
        return 0
    current_depth = 1
    queue = [(head, current_depth)]
    while queue and (max_depth is None or current_depth < max_depth):
        e, depth = queue.pop(0)
        current_depth = max(current_depth, depth)
        cmt = self[e]
        if isinstance(cmt, Tag):
            _cls, sha = cmt.object
            cmt = self[sha]
        queue.extend(((parent, depth + 1) for parent in get_parents(cmt) if parent in self))
    return current_depth