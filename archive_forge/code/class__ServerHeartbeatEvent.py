from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class _ServerHeartbeatEvent:
    """Base class for server heartbeat events."""
    __slots__ = ('__connection_id', '__awaited')

    def __init__(self, connection_id: _Address, awaited: bool=False) -> None:
        self.__connection_id = connection_id
        self.__awaited = awaited

    @property
    def connection_id(self) -> _Address:
        """The address (host, port) of the server this heartbeat was sent
        to.
        """
        return self.__connection_id

    @property
    def awaited(self) -> bool:
        """Whether the heartbeat was issued as an awaitable hello command.

        .. versionadded:: 4.6
        """
        return self.__awaited

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.connection_id} awaited: {self.awaited}>'