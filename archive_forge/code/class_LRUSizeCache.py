from typing import Callable, Dict, Generic, Iterable, Iterator, Optional, TypeVar
class LRUSizeCache(LRUCache[K, V]):
    """An LRUCache that removes things based on the size of the values.

    This differs in that it doesn't care how many actual items there are,
    it just restricts the cache to be cleaned up after so much data is stored.

    The size of items added will be computed using compute_size(value), which
    defaults to len() if not supplied.
    """
    _compute_size: Callable[[V], int]

    def __init__(self, max_size: int=1024 * 1024, after_cleanup_size: Optional[int]=None, compute_size: Optional[Callable[[V], int]]=None) -> None:
        """Create a new LRUSizeCache.

        Args:
          max_size: The max number of bytes to store before we start
            clearing out entries.
          after_cleanup_size: After cleaning up, shrink everything to this
            size.
          compute_size: A function to compute the size of the values. We
            use a function here, so that you can pass 'len' if you are just
            using simple strings, or a more complex function if you are using
            something like a list of strings, or even a custom object.
            The function should take the form "compute_size(value) => integer".
            If not supplied, it defaults to 'len()'
        """
        self._value_size = 0
        if compute_size is None:
            self._compute_size = len
        else:
            self._compute_size = compute_size
        self._update_max_size(max_size, after_cleanup_size=after_cleanup_size)
        LRUCache.__init__(self, max_cache=max(int(max_size / 512), 1))

    def add(self, key: K, value: V, cleanup: Optional[Callable[[K, V], None]]=None) -> None:
        """Add a new value to the cache.

        Also, if the entry is ever removed from the cache, call
        cleanup(key, value).

        Args:
          key: The key to store it under
          value: The object to store
          cleanup: None or a function taking (key, value) to indicate
                        'value' should be cleaned up.
        """
        if key is _null_key:
            raise ValueError('cannot use _null_key as a key')
        node = self._cache.get(key, None)
        value_len = self._compute_size(value)
        if value_len >= self._after_cleanup_size:
            if node is not None:
                self._remove_node(node)
            if cleanup is not None:
                cleanup(key, value)
            return
        if node is None:
            node = _LRUNode(key, value, cleanup=cleanup)
            self._cache[key] = node
        else:
            assert node.size is not None
            self._value_size -= node.size
        node.size = value_len
        self._value_size += value_len
        self._record_access(node)
        if self._value_size > self._max_size:
            self.cleanup()

    def cleanup(self) -> None:
        """Clear the cache until it shrinks to the requested size.

        This does not completely wipe the cache, just makes sure it is under
        the after_cleanup_size.
        """
        while self._value_size > self._after_cleanup_size:
            self._remove_lru()

    def _remove_node(self, node: _LRUNode[K, V]) -> None:
        assert node.size is not None
        self._value_size -= node.size
        LRUCache._remove_node(self, node)

    def resize(self, max_size: int, after_cleanup_size: Optional[int]=None) -> None:
        """Change the number of bytes that will be cached."""
        self._update_max_size(max_size, after_cleanup_size=after_cleanup_size)
        max_cache = max(int(max_size / 512), 1)
        self._update_max_cache(max_cache)

    def _update_max_size(self, max_size: int, after_cleanup_size: Optional[int]=None) -> None:
        self._max_size = max_size
        if after_cleanup_size is None:
            self._after_cleanup_size = self._max_size * 8 // 10
        else:
            self._after_cleanup_size = min(after_cleanup_size, self._max_size)