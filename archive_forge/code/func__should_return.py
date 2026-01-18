import collections
import heapq
from itertools import chain
from typing import Deque, Dict, List, Optional, Set, Tuple
from .diff_tree import (
from .errors import MissingCommitError
from .objects import Commit, ObjectID, Tag
def _should_return(self, entry):
    """Determine if a walk entry should be returned..

        Args:
          entry: The WalkEntry to consider.
        Returns: True if the WalkEntry should be returned by this walk, or
            False otherwise (e.g. if it doesn't match any requested paths).
        """
    commit = entry.commit
    if self.since is not None and commit.commit_time < self.since:
        return False
    if self.until is not None and commit.commit_time > self.until:
        return False
    if commit.id in self.excluded:
        return False
    if self.paths is None:
        return True
    if len(self.get_parents(commit)) > 1:
        for path_changes in entry.changes():
            for change in path_changes:
                if self._change_matches(change):
                    return True
    else:
        for change in entry.changes():
            if self._change_matches(change):
                return True
    return None