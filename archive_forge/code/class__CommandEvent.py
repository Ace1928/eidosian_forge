from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class _CommandEvent:
    """Base class for command events."""
    __slots__ = ('__cmd_name', '__rqst_id', '__conn_id', '__op_id', '__service_id', '__db')

    def __init__(self, command_name: str, request_id: int, connection_id: _Address, operation_id: Optional[int], service_id: Optional[ObjectId]=None, database_name: str='') -> None:
        self.__cmd_name = command_name
        self.__rqst_id = request_id
        self.__conn_id = connection_id
        self.__op_id = operation_id
        self.__service_id = service_id
        self.__db = database_name

    @property
    def command_name(self) -> str:
        """The command name."""
        return self.__cmd_name

    @property
    def request_id(self) -> int:
        """The request id for this operation."""
        return self.__rqst_id

    @property
    def connection_id(self) -> _Address:
        """The address (host, port) of the server this command was sent to."""
        return self.__conn_id

    @property
    def service_id(self) -> Optional[ObjectId]:
        """The service_id this command was sent to, or ``None``.

        .. versionadded:: 3.12
        """
        return self.__service_id

    @property
    def operation_id(self) -> Optional[int]:
        """An id for this series of events or None."""
        return self.__op_id

    @property
    def database_name(self) -> str:
        """The database_name this command was sent to, or ``""``.

        .. versionadded:: 4.6
        """
        return self.__db