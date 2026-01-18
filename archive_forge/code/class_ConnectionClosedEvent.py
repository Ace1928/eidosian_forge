from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class ConnectionClosedEvent(_ConnectionIdEvent):
    """Published when a Connection is closed.

    :Parameters:
     - `address`: The address (host, port) pair of the server this
       Connection is attempting to connect to.
     - `connection_id`: The integer ID of the Connection in this Pool.
     - `reason`: A reason explaining why this connection was closed.

    .. versionadded:: 3.9
    """
    __slots__ = ('__reason',)

    def __init__(self, address: _Address, connection_id: int, reason: str):
        super().__init__(address, connection_id)
        self.__reason = reason

    @property
    def reason(self) -> str:
        """A reason explaining why this connection was closed.

        The reason must be one of the strings from the
        :class:`ConnectionClosedReason` enum.
        """
        return self.__reason

    def __repr__(self) -> str:
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__, self.address, self.connection_id, self.__reason)