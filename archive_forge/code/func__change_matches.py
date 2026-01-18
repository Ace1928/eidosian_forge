import collections
import heapq
from itertools import chain
from typing import Deque, Dict, List, Optional, Set, Tuple
from .diff_tree import (
from .errors import MissingCommitError
from .objects import Commit, ObjectID, Tag
def _change_matches(self, change):
    if not change:
        return False
    old_path = change.old.path
    new_path = change.new.path
    if self._path_matches(new_path):
        if self.follow and change.type in RENAME_CHANGE_TYPES:
            self.paths.add(old_path)
            self.paths.remove(new_path)
        return True
    elif self._path_matches(old_path):
        return True
    return False