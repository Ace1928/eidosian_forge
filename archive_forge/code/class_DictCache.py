from __future__ import annotations
from threading import Lock
from typing import IO, TYPE_CHECKING, MutableMapping
class DictCache(BaseCache):

    def __init__(self, init_dict: MutableMapping[str, bytes] | None=None) -> None:
        self.lock = Lock()
        self.data = init_dict or {}

    def get(self, key: str) -> bytes | None:
        return self.data.get(key, None)

    def set(self, key: str, value: bytes, expires: int | datetime | None=None) -> None:
        with self.lock:
            self.data.update({key: value})

    def delete(self, key: str) -> None:
        with self.lock:
            if key in self.data:
                self.data.pop(key)