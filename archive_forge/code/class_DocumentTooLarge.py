from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class DocumentTooLarge(InvalidDocument):
    """Raised when an encoded document is too large for the connected server."""