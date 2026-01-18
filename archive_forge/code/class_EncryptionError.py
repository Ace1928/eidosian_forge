from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class EncryptionError(PyMongoError):
    """Raised when encryption or decryption fails.

    This error always wraps another exception which can be retrieved via the
    :attr:`cause` property.

    .. versionadded:: 3.9
    """

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.__cause = cause

    @property
    def cause(self) -> Exception:
        """The exception that caused this encryption or decryption error."""
        return self.__cause

    @property
    def timeout(self) -> bool:
        if isinstance(self.__cause, PyMongoError):
            return self.__cause.timeout
        return False