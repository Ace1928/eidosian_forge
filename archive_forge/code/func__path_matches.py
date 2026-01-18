import collections
import heapq
from itertools import chain
from typing import Deque, Dict, List, Optional, Set, Tuple
from .diff_tree import (
from .errors import MissingCommitError
from .objects import Commit, ObjectID, Tag
def _path_matches(self, changed_path):
    if changed_path is None:
        return False
    if self.paths is None:
        return True
    for followed_path in self.paths:
        if changed_path == followed_path:
            return True
        if changed_path.startswith(followed_path) and changed_path[len(followed_path)] == b'/'[0]:
            return True
    return False