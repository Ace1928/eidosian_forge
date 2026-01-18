from __future__ import annotations
import itertools
from abc import abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable
@runtime_checkable
class CacheStatsProvider(Protocol):

    @abstractmethod
    def get_stats(self) -> list[CacheStat]:
        raise NotImplementedError