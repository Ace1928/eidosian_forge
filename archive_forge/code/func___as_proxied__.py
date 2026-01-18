from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Iterable, cast
from typing_extensions import override
def __as_proxied__(self) -> T:
    """Helper method that returns the current proxy, typed as the loaded object"""
    return cast(T, self)