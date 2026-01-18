from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class ConnectionCheckOutFailedEvent(_ConnectionEvent):
    """Published when the driver's attempt to check out a connection fails.

    :Parameters:
     - `address`: The address (host, port) pair of the server this
       Connection is attempting to connect to.
     - `reason`: A reason explaining why connection check out failed.

    .. versionadded:: 3.9
    """
    __slots__ = ('__reason',)

    def __init__(self, address: _Address, reason: str) -> None:
        super().__init__(address)
        self.__reason = reason

    @property
    def reason(self) -> str:
        """A reason explaining why connection check out failed.

        The reason must be one of the strings from the
        :class:`ConnectionCheckOutFailedReason` enum.
        """
        return self.__reason

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.address!r}, {self.__reason!r})'