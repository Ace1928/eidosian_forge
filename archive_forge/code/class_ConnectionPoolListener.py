from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class ConnectionPoolListener(_EventListener):
    """Abstract base class for connection pool listeners.

    Handles all of the connection pool events defined in the Connection
    Monitoring and Pooling Specification:
    :class:`PoolCreatedEvent`, :class:`PoolClearedEvent`,
    :class:`PoolClosedEvent`, :class:`ConnectionCreatedEvent`,
    :class:`ConnectionReadyEvent`, :class:`ConnectionClosedEvent`,
    :class:`ConnectionCheckOutStartedEvent`,
    :class:`ConnectionCheckOutFailedEvent`,
    :class:`ConnectionCheckedOutEvent`,
    and :class:`ConnectionCheckedInEvent`.

    .. versionadded:: 3.9
    """

    def pool_created(self, event: PoolCreatedEvent) -> None:
        """Abstract method to handle a :class:`PoolCreatedEvent`.

        Emitted when a connection Pool is created.

        :Parameters:
          - `event`: An instance of :class:`PoolCreatedEvent`.
        """
        raise NotImplementedError

    def pool_ready(self, event: PoolReadyEvent) -> None:
        """Abstract method to handle a :class:`PoolReadyEvent`.

        Emitted when a connection Pool is marked ready.

        :Parameters:
          - `event`: An instance of :class:`PoolReadyEvent`.

        .. versionadded:: 4.0
        """
        raise NotImplementedError

    def pool_cleared(self, event: PoolClearedEvent) -> None:
        """Abstract method to handle a `PoolClearedEvent`.

        Emitted when a connection Pool is cleared.

        :Parameters:
          - `event`: An instance of :class:`PoolClearedEvent`.
        """
        raise NotImplementedError

    def pool_closed(self, event: PoolClosedEvent) -> None:
        """Abstract method to handle a `PoolClosedEvent`.

        Emitted when a connection Pool is closed.

        :Parameters:
          - `event`: An instance of :class:`PoolClosedEvent`.
        """
        raise NotImplementedError

    def connection_created(self, event: ConnectionCreatedEvent) -> None:
        """Abstract method to handle a :class:`ConnectionCreatedEvent`.

        Emitted when a connection Pool creates a Connection object.

        :Parameters:
          - `event`: An instance of :class:`ConnectionCreatedEvent`.
        """
        raise NotImplementedError

    def connection_ready(self, event: ConnectionReadyEvent) -> None:
        """Abstract method to handle a :class:`ConnectionReadyEvent`.

        Emitted when a connection has finished its setup, and is now ready to
        use.

        :Parameters:
          - `event`: An instance of :class:`ConnectionReadyEvent`.
        """
        raise NotImplementedError

    def connection_closed(self, event: ConnectionClosedEvent) -> None:
        """Abstract method to handle a :class:`ConnectionClosedEvent`.

        Emitted when a connection Pool closes a connection.

        :Parameters:
          - `event`: An instance of :class:`ConnectionClosedEvent`.
        """
        raise NotImplementedError

    def connection_check_out_started(self, event: ConnectionCheckOutStartedEvent) -> None:
        """Abstract method to handle a :class:`ConnectionCheckOutStartedEvent`.

        Emitted when the driver starts attempting to check out a connection.

        :Parameters:
          - `event`: An instance of :class:`ConnectionCheckOutStartedEvent`.
        """
        raise NotImplementedError

    def connection_check_out_failed(self, event: ConnectionCheckOutFailedEvent) -> None:
        """Abstract method to handle a :class:`ConnectionCheckOutFailedEvent`.

        Emitted when the driver's attempt to check out a connection fails.

        :Parameters:
          - `event`: An instance of :class:`ConnectionCheckOutFailedEvent`.
        """
        raise NotImplementedError

    def connection_checked_out(self, event: ConnectionCheckedOutEvent) -> None:
        """Abstract method to handle a :class:`ConnectionCheckedOutEvent`.

        Emitted when the driver successfully checks out a connection.

        :Parameters:
          - `event`: An instance of :class:`ConnectionCheckedOutEvent`.
        """
        raise NotImplementedError

    def connection_checked_in(self, event: ConnectionCheckedInEvent) -> None:
        """Abstract method to handle a :class:`ConnectionCheckedInEvent`.

        Emitted when the driver checks in a connection back to the connection
        Pool.

        :Parameters:
          - `event`: An instance of :class:`ConnectionCheckedInEvent`.
        """
        raise NotImplementedError