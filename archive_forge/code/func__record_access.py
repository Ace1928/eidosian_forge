from typing import Callable, Dict, Generic, Iterable, Iterator, Optional, TypeVar
def _record_access(self, node: _LRUNode[K, V]) -> None:
    """Record that key was accessed."""
    if self._most_recently_used is None:
        self._most_recently_used = node
        self._least_recently_used = node
        return
    elif node is self._most_recently_used:
        return
    if node is self._least_recently_used:
        self._least_recently_used = node.prev
    if node.prev is not None:
        node.prev.next_key = node.next_key
    if node.next_key is not _null_key:
        node_next = self._cache[node.next_key]
        node_next.prev = node.prev
    node.next_key = self._most_recently_used.key
    self._most_recently_used.prev = node
    self._most_recently_used = node
    node.prev = None