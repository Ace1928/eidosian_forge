from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class _ConnectionIdEvent(_ConnectionEvent):
    """Private base class for connection events with an id."""
    __slots__ = ('__connection_id',)

    def __init__(self, address: _Address, connection_id: int) -> None:
        super().__init__(address)
        self.__connection_id = connection_id

    @property
    def connection_id(self) -> int:
        """The ID of the connection."""
        return self.__connection_id

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.address!r}, {self.__connection_id!r})'