import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _process_next_old(self):
    refs = self._old_queue
    self._old_queue = []
    all_old_chks = self._all_old_chks
    for record, _, prefix_refs, items in self._read_nodes_from_store(refs):
        self._all_old_items.update(items)
        refs = [r for _, r in prefix_refs if r not in all_old_chks]
        self._old_queue.extend(refs)
        all_old_chks.update(refs)