import collections
import heapq
from itertools import chain
from typing import Deque, Dict, List, Optional, Set, Tuple
from .diff_tree import (
from .errors import MissingCommitError
from .objects import Commit, ObjectID, Tag
def _reorder(self, results):
    """Possibly reorder a results iterator.

        Args:
          results: An iterator of WalkEntry objects, in the order returned
            from the queue_cls.
        Returns: An iterator or list of WalkEntry objects, in the order
            required by the Walker.
        """
    if self.order == ORDER_TOPO:
        results = _topo_reorder(results, self.get_parents)
    if self.reverse:
        results = reversed(list(results))
    return results