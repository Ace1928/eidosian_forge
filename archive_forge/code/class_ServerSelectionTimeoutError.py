from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class ServerSelectionTimeoutError(AutoReconnect):
    """Thrown when no MongoDB server is available for an operation

    If there is no suitable server for an operation PyMongo tries for
    ``serverSelectionTimeoutMS`` (default 30 seconds) to find one, then
    throws this exception. For example, it is thrown after attempting an
    operation when PyMongo cannot connect to any server, or if you attempt
    an insert into a replica set that has no primary and does not elect one
    within the timeout window, or if you attempt to query with a Read
    Preference that the replica set cannot satisfy.
    """

    @property
    def timeout(self) -> bool:
        return True