from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class ConnectionCheckedOutEvent(_ConnectionIdEvent):
    """Published when the driver successfully checks out a connection.

    :Parameters:
     - `address`: The address (host, port) pair of the server this
       Connection is attempting to connect to.
     - `connection_id`: The integer ID of the Connection in this Pool.

    .. versionadded:: 3.9
    """
    __slots__ = ()