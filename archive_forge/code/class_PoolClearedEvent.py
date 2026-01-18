from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class PoolClearedEvent(_PoolEvent):
    """Published when a Connection Pool is cleared.

    :Parameters:
     - `address`: The address (host, port) pair of the server this Pool is
       attempting to connect to.
     - `service_id`: The service_id this command was sent to, or ``None``.

    .. versionadded:: 3.9
    """
    __slots__ = ('__service_id',)

    def __init__(self, address: _Address, service_id: Optional[ObjectId]=None) -> None:
        super().__init__(address)
        self.__service_id = service_id

    @property
    def service_id(self) -> Optional[ObjectId]:
        """Connections with this service_id are cleared.

        When service_id is ``None``, all connections in the pool are cleared.

        .. versionadded:: 3.12
        """
        return self.__service_id

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.address!r}, {self.__service_id!r})'