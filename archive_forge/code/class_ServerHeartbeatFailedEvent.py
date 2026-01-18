from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class ServerHeartbeatFailedEvent(_ServerHeartbeatEvent):
    """Fired when the server heartbeat fails, either with an "ok: 0"
    or a socket exception.

    .. versionadded:: 3.3
    """
    __slots__ = ('__duration', '__reply')

    def __init__(self, duration: float, reply: Exception, connection_id: _Address, awaited: bool=False) -> None:
        super().__init__(connection_id, awaited)
        self.__duration = duration
        self.__reply = reply

    @property
    def duration(self) -> float:
        """The duration of this heartbeat in microseconds."""
        return self.__duration

    @property
    def reply(self) -> Exception:
        """A subclass of :exc:`Exception`."""
        return self.__reply

    @property
    def awaited(self) -> bool:
        """Whether the heartbeat was awaited.

        If true, then :meth:`duration` reflects the sum of the round trip time
        to the server and the time that the server waited before sending a
        response.

        .. versionadded:: 3.11
        """
        return super().awaited

    def __repr__(self) -> str:
        return '<{} {} duration: {}, awaited: {}, reply: {!r}>'.format(self.__class__.__name__, self.connection_id, self.duration, self.awaited, self.reply)