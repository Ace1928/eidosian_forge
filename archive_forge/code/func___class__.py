from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Iterable, cast
from typing_extensions import override
@property
@override
def __class__(self) -> type:
    proxied = self.__get_proxied__()
    if issubclass(type(proxied), LazyProxy):
        return type(proxied)
    return proxied.__class__