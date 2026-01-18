from __future__ import annotations
from pymongo.errors import PyMongoError
class NoFile(GridFSError):
    """Raised when trying to read from a non-existent file."""