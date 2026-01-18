from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class NotPrimaryError(AutoReconnect):
    """The server responded "not primary" or "node is recovering".

    These errors result from a query, write, or command. The operation failed
    because the client thought it was using the primary but the primary has
    stepped down, or the client thought it was using a healthy secondary but
    the secondary is stale and trying to recover.

    The client launches a refresh operation on a background thread, to update
    its view of the server as soon as possible after throwing this exception.

    Subclass of :exc:`~pymongo.errors.AutoReconnect`.

    .. versionadded:: 3.12
    """

    def __init__(self, message: str='', errors: Optional[Union[Mapping[str, Any], list[Any]]]=None) -> None:
        super().__init__(_format_detailed_error(message, errors), errors=errors)