from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class ServerHeartbeatListener(_EventListener):
    """Abstract base class for server heartbeat listeners.

    Handles `ServerHeartbeatStartedEvent`, `ServerHeartbeatSucceededEvent`,
    and `ServerHeartbeatFailedEvent`.

    .. versionadded:: 3.3
    """

    def started(self, event: ServerHeartbeatStartedEvent) -> None:
        """Abstract method to handle a `ServerHeartbeatStartedEvent`.

        :Parameters:
          - `event`: An instance of :class:`ServerHeartbeatStartedEvent`.
        """
        raise NotImplementedError

    def succeeded(self, event: ServerHeartbeatSucceededEvent) -> None:
        """Abstract method to handle a `ServerHeartbeatSucceededEvent`.

        :Parameters:
          - `event`: An instance of :class:`ServerHeartbeatSucceededEvent`.
        """
        raise NotImplementedError

    def failed(self, event: ServerHeartbeatFailedEvent) -> None:
        """Abstract method to handle a `ServerHeartbeatFailedEvent`.

        :Parameters:
          - `event`: An instance of :class:`ServerHeartbeatFailedEvent`.
        """
        raise NotImplementedError