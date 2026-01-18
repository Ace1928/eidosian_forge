from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class CursorNotFound(OperationFailure):
    """Raised while iterating query results if the cursor is
    invalidated on the server.

    .. versionadded:: 2.7
    """