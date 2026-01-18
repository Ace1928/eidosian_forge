from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
class ConnectionFailure(PyMongoError):
    """Raised when a connection to the database cannot be made or is lost."""