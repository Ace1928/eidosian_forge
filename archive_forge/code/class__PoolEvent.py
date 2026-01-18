from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class _PoolEvent:
    """Base class for pool events."""
    __slots__ = ('__address',)

    def __init__(self, address: _Address) -> None:
        self.__address = address

    @property
    def address(self) -> _Address:
        """The address (host, port) pair of the server the pool is attempting
        to connect to.
        """
        return self.__address

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.__address!r})'