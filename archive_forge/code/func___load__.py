from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Iterable, cast
from typing_extensions import override
@abstractmethod
def __load__(self) -> T:
    ...