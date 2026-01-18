from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class PyMongoError(Exception):
    """Base class for all PyMongo exceptions."""

    def __init__(self, message: str='', error_labels: Optional[Iterable[str]]=None) -> None:
        super().__init__(message)
        self._message = message
        self._error_labels = set(error_labels or [])

    def has_error_label(self, label: str) -> bool:
        """Return True if this error contains the given label.

        .. versionadded:: 3.7
        """
        return label in self._error_labels

    def _add_error_label(self, label: str) -> None:
        """Add the given label to this error."""
        self._error_labels.add(label)

    def _remove_error_label(self, label: str) -> None:
        """Remove the given label from this error."""
        self._error_labels.discard(label)

    @property
    def timeout(self) -> bool:
        """True if this error was caused by a timeout.

        .. versionadded:: 4.2
        """
        return False