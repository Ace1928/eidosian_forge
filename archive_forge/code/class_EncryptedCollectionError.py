from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class EncryptedCollectionError(EncryptionError):
    """Raised when creating a collection with encrypted_fields fails.

    .. versionadded:: 4.4
    """

    def __init__(self, cause: Exception, encrypted_fields: Mapping[str, Any]) -> None:
        super().__init__(cause)
        self.__encrypted_fields = encrypted_fields

    @property
    def encrypted_fields(self) -> Mapping[str, Any]:
        """The encrypted_fields document that allows inferring which data keys are *known* to be created.

        Note that the returned document is not guaranteed to contain information about *all* of the data keys that
        were created, for example in the case of an indefinite error like a timeout. Use the `cause` property to
        determine whether a definite or indefinite error caused this error, and only rely on the accuracy of the
        encrypted_fields if the error is definite.
        """
        return self.__encrypted_fields