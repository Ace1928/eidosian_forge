from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class PoolClosedEvent(_PoolEvent):
    """Published when a Connection Pool is closed.

    :Parameters:
     - `address`: The address (host, port) pair of the server this Pool is
       attempting to connect to.

    .. versionadded:: 3.9
    """
    __slots__ = ()