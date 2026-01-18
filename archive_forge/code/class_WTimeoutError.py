from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class WTimeoutError(WriteConcernError):
    """Raised when a database operation times out (i.e. wtimeout expires)
    before replication completes.

    With newer versions of MongoDB the `details` attribute may include
    write concern fields like 'n', 'updatedExisting', or 'writtenTo'.

    .. versionadded:: 2.7
    """

    @property
    def timeout(self) -> bool:
        return True