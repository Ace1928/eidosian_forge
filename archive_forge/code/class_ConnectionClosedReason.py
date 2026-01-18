from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class ConnectionClosedReason:
    """An enum that defines values for `reason` on a
    :class:`ConnectionClosedEvent`.

    .. versionadded:: 3.9
    """
    STALE = 'stale'
    'The pool was cleared, making the connection no longer valid.'
    IDLE = 'idle'
    'The connection became stale by being idle for too long (maxIdleTimeMS).\n    '
    ERROR = 'error'
    'The connection experienced an error, making it no longer valid.'
    POOL_CLOSED = 'poolClosed'
    'The pool was closed, making the connection no longer valid.'