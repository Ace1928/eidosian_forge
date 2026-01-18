from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class AutoReconnect(ConnectionFailure):
    """Raised when a connection to the database is lost and an attempt to
    auto-reconnect will be made.

    In order to auto-reconnect you must handle this exception, recognizing that
    the operation which caused it has not necessarily succeeded. Future
    operations will attempt to open a new connection to the database (and
    will continue to raise this exception until the first successful
    connection is made).

    Subclass of :exc:`~pymongo.errors.ConnectionFailure`.
    """
    errors: Union[Mapping[str, Any], Sequence[Any]]
    details: Union[Mapping[str, Any], Sequence[Any]]

    def __init__(self, message: str='', errors: Optional[Union[Mapping[str, Any], Sequence[Any]]]=None) -> None:
        error_labels = None
        if errors is not None:
            if isinstance(errors, dict):
                error_labels = errors.get('errorLabels')
        super().__init__(message, error_labels)
        self.errors = self.details = errors or []