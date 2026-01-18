from __future__ import annotations
from threading import Lock
from typing import IO, TYPE_CHECKING, MutableMapping
class SeparateBodyBaseCache(BaseCache):
    """
    In this variant, the body is not stored mixed in with the metadata, but is
    passed in (as a bytes-like object) in a separate call to ``set_body()``.

    That is, the expected interaction pattern is::

        cache.set(key, serialized_metadata)
        cache.set_body(key)

    Similarly, the body should be loaded separately via ``get_body()``.
    """

    def set_body(self, key: str, body: bytes) -> None:
        raise NotImplementedError()

    def get_body(self, key: str) -> IO[bytes] | None:
        """
        Return the body as file-like object.
        """
        raise NotImplementedError()