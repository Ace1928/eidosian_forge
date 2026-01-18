import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _read_old_roots(self):
    old_chks_to_enqueue = []
    all_old_chks = self._all_old_chks
    for record, node, prefix_refs, items in self._read_nodes_from_store(self._old_root_keys):
        prefix_refs = [p_r for p_r in prefix_refs if p_r[1] not in all_old_chks]
        new_refs = [p_r[1] for p_r in prefix_refs]
        all_old_chks.update(new_refs)
        self._all_old_items.update(items)
        old_chks_to_enqueue.extend(prefix_refs)
    return old_chks_to_enqueue