from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol
class InvalidCacheStorageContext(CacheStorageError):
    """Raised if the cache storage manager is not able to work with
    provided CacheStorageContext.
    """