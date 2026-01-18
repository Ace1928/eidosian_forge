from _weakref import (
from _weakrefset import WeakSet, _IterationGuard
import _collections_abc  # Import after _weakref to avoid circular import.
import sys
import itertools
def _commit_removals(self):
    pop = self._pending_removals.pop
    d = self.data
    while True:
        try:
            key = pop()
        except IndexError:
            return
        try:
            del d[key]
        except KeyError:
            pass