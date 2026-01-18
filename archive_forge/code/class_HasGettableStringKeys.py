from __future__ import annotations
import typing
from collections import OrderedDict
from enum import Enum, auto
from threading import RLock
class HasGettableStringKeys(Protocol):

    def keys(self) -> typing.Iterator[str]:
        ...

    def __getitem__(self, key: str) -> str:
        ...