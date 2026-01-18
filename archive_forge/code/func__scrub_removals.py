from _weakref import (
from _weakrefset import WeakSet, _IterationGuard
import _collections_abc  # Import after _weakref to avoid circular import.
import sys
import itertools
def _scrub_removals(self):
    d = self.data
    self._pending_removals = [k for k in self._pending_removals if k in d]
    self._dirty_len = False