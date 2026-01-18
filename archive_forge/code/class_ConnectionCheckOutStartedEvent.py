from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class ConnectionCheckOutStartedEvent(_ConnectionEvent):
    """Published when the driver starts attempting to check out a connection.

    :Parameters:
     - `address`: The address (host, port) pair of the server this
       Connection is attempting to connect to.

    .. versionadded:: 3.9
    """
    __slots__ = ()