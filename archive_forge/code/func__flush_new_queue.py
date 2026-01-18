import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _flush_new_queue(self):
    refs = set(self._new_queue)
    self._new_queue = []
    all_old_chks = self._all_old_chks
    processed_new_refs = self._processed_new_refs
    all_old_items = self._all_old_items
    new_items = [item for item in self._new_item_queue if item not in all_old_items]
    self._new_item_queue = []
    if new_items:
        yield (None, new_items)
    refs = refs.difference(all_old_chks)
    processed_new_refs.update(refs)
    while refs:
        next_refs = set()
        next_refs_update = next_refs.update
        for record, _, p_refs, items in self._read_nodes_from_store(refs):
            if all_old_items:
                items = [item for item in items if item not in all_old_items]
            yield (record, items)
            next_refs_update([p_r[1] for p_r in p_refs])
            del p_refs
        next_refs = next_refs.difference(all_old_chks)
        next_refs = next_refs.difference(processed_new_refs)
        processed_new_refs.update(next_refs)
        refs = next_refs