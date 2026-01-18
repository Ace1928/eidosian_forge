from __future__ import annotations
import logging
from pymongo import monitoring
A simple listener that logs server connection pool events.

    Listens for :class:`~pymongo.monitoring.PoolCreatedEvent`,
    :class:`~pymongo.monitoring.PoolClearedEvent`,
    :class:`~pymongo.monitoring.PoolClosedEvent`,
    :~pymongo.monitoring.class:`ConnectionCreatedEvent`,
    :class:`~pymongo.monitoring.ConnectionReadyEvent`,
    :class:`~pymongo.monitoring.ConnectionClosedEvent`,
    :class:`~pymongo.monitoring.ConnectionCheckOutStartedEvent`,
    :class:`~pymongo.monitoring.ConnectionCheckOutFailedEvent`,
    :class:`~pymongo.monitoring.ConnectionCheckedOutEvent`,
    and :class:`~pymongo.monitoring.ConnectionCheckedInEvent`
    events and logs them at the `INFO` severity level using :mod:`logging`.

    .. versionadded:: 3.11
    