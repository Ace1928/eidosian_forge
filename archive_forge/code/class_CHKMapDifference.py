import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
class CHKMapDifference:
    """Iterate the stored pages and key,value pairs for (new - old).

    This class provides a generator over the stored CHK pages and the
    (key, value) pairs that are in any of the new maps and not in any of the
    old maps.

    Note that it may yield chk pages that are common (especially root nodes),
    but it won't yield (key,value) pairs that are common.
    """

    def __init__(self, store, new_root_keys, old_root_keys, search_key_func, pb=None):
        self._store = store
        self._new_root_keys = new_root_keys
        self._old_root_keys = old_root_keys
        self._pb = pb
        self._all_old_chks = set(self._old_root_keys)
        self._all_old_items = set()
        self._processed_new_refs = set()
        self._search_key_func = search_key_func
        self._old_queue = []
        self._new_queue = []
        self._new_item_queue = []
        self._state = None

    def _read_nodes_from_store(self, keys):
        as_st = StaticTuple.from_sequence
        stream = self._store.get_record_stream(keys, 'unordered', True)
        for record in stream:
            if self._pb is not None:
                self._pb.tick()
            if record.storage_kind == 'absent':
                raise errors.NoSuchRevision(self._store, record.key)
            bytes = record.get_bytes_as('fulltext')
            node = _deserialise(bytes, record.key, search_key_func=self._search_key_func)
            if isinstance(node, InternalNode):
                prefix_refs = list(node._items.items())
                items = []
            else:
                prefix_refs = []
                items = list(node._items.items())
            yield (record, node, prefix_refs, items)

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

    def _enqueue_old(self, new_prefixes, old_chks_to_enqueue):
        for prefix, ref in old_chks_to_enqueue:
            not_interesting = True
            for i in range(len(prefix), 0, -1):
                if prefix[:i] in new_prefixes:
                    not_interesting = False
                    break
            if not_interesting:
                continue
            self._old_queue.append(ref)

    def _read_all_roots(self):
        """Read the root pages.

        This is structured as a generator, so that the root records can be
        yielded up to whoever needs them without any buffering.
        """
        if not self._old_root_keys:
            self._new_queue = list(self._new_root_keys)
            return
        old_chks_to_enqueue = self._read_old_roots()
        new_keys = set(self._new_root_keys).difference(self._all_old_chks)
        new_prefixes = set()
        processed_new_refs = self._processed_new_refs
        processed_new_refs.update(new_keys)
        for record, node, prefix_refs, items in self._read_nodes_from_store(new_keys):
            prefix_refs = [p_r for p_r in prefix_refs if p_r[1] not in self._all_old_chks and p_r[1] not in processed_new_refs]
            refs = [p_r[1] for p_r in prefix_refs]
            new_prefixes.update([p_r[0] for p_r in prefix_refs])
            self._new_queue.extend(refs)
            new_items = [item for item in items if item not in self._all_old_items]
            self._new_item_queue.extend(new_items)
            new_prefixes.update([self._search_key_func(item[0]) for item in new_items])
            processed_new_refs.update(refs)
            yield record
        for prefix in list(new_prefixes):
            new_prefixes.update([prefix[:i] for i in range(1, len(prefix))])
        self._enqueue_old(new_prefixes, old_chks_to_enqueue)

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

    def _process_next_old(self):
        refs = self._old_queue
        self._old_queue = []
        all_old_chks = self._all_old_chks
        for record, _, prefix_refs, items in self._read_nodes_from_store(refs):
            self._all_old_items.update(items)
            refs = [r for _, r in prefix_refs if r not in all_old_chks]
            self._old_queue.extend(refs)
            all_old_chks.update(refs)

    def _process_queues(self):
        while self._old_queue:
            self._process_next_old()
        return self._flush_new_queue()

    def process(self):
        for record in self._read_all_roots():
            yield (record, [])
        for record, items in self._process_queues():
            yield (record, items)