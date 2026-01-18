from __future__ import annotations
import hashlib
import os
from textwrap import dedent
from typing import IO, TYPE_CHECKING
from pip._vendor.cachecontrol.cache import BaseCache, SeparateBodyBaseCache
from pip._vendor.cachecontrol.controller import CacheController
class SeparateBodyFileCache(_FileCacheMixin, SeparateBodyBaseCache):
    """
    Memory-efficient FileCache: body is stored in a separate file, reducing
    peak memory usage.
    """

    def get_body(self, key: str) -> IO[bytes] | None:
        name = self._fn(key) + '.body'
        try:
            return open(name, 'rb')
        except FileNotFoundError:
            return None

    def set_body(self, key: str, body: bytes) -> None:
        name = self._fn(key) + '.body'
        self._write(name, body)

    def delete(self, key: str) -> None:
        self._delete(key, '')
        self._delete(key, '.body')