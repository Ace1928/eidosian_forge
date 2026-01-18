from __future__ import annotations
from collections import deque
import dataclasses
from enum import Enum
import threading
import time
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from .. import event
from .. import exc
from .. import log
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
class _AdhocProxiedConnection(PoolProxiedConnection):
    """provides the :class:`.PoolProxiedConnection` interface for cases where
    the DBAPI connection is not actually proxied.

    This is used by the engine internals to pass a consistent
    :class:`.PoolProxiedConnection` object to consuming dialects in response to
    pool events that may not always have the :class:`._ConnectionFairy`
    available.

    """
    __slots__ = ('dbapi_connection', '_connection_record', '_is_valid')
    dbapi_connection: DBAPIConnection
    _connection_record: ConnectionPoolEntry

    def __init__(self, dbapi_connection: DBAPIConnection, connection_record: ConnectionPoolEntry):
        self.dbapi_connection = dbapi_connection
        self._connection_record = connection_record
        self._is_valid = True

    @property
    def driver_connection(self) -> Any:
        return self._connection_record.driver_connection

    @property
    def connection(self) -> DBAPIConnection:
        return self.dbapi_connection

    @property
    def is_valid(self) -> bool:
        """Implement is_valid state attribute.

        for the adhoc proxied connection it's assumed the connection is valid
        as there is no "invalidate" routine.

        """
        return self._is_valid

    def invalidate(self, e: Optional[BaseException]=None, soft: bool=False) -> None:
        self._is_valid = False

    @util.ro_non_memoized_property
    def record_info(self) -> Optional[_InfoType]:
        return self._connection_record.record_info

    def cursor(self, *args: Any, **kwargs: Any) -> DBAPICursor:
        return self.dbapi_connection.cursor(*args, **kwargs)

    def __getattr__(self, key: Any) -> Any:
        return getattr(self.dbapi_connection, key)